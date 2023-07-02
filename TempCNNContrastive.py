import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
from torch.nn.modules import LayerNorm, Linear, Sequential, ReLU
import torch.utils.data
import os
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator
import argparse
import itertools
from torch import nn
from torch.nn import functional as F
#from torchsampler import ImbalancedDatasetSampler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torchmetrics
import copy
from glob import glob
from utils import rasterize
from plot_confusion_report import *
import itertools
import copy
from glob import glob
from utils import rasterize
from plot_confusion_report import *
import itertools

def OneHot(x):
    x = list(x)
    x = torch.as_tensor(x).long()
    x = nn.functional.one_hot(x)
    return x

class CCE(nn.Module):
    def __init__(self, reduction='mean', epsilon=0.0000001):
        # self.with_logits = with_logits
        # self.class_weight = class_weight
        self.reduction = reduction
        self.epsilon =  epsilon
        super(CCE, self).__init__()
        
    def forward(self, pred, ref, c_weight):
        # if pred.shape != ref.shape:
        #     raise ValueError('The categorical cross entropy is expecting arrays of equal size')
        # if len(ref.shape) == 1:
        #     print('The CCE loss is expecing the predicted values as one-hot encoded vectors, changes will be applied')
        #     ypred = self.onehot(ypred)
        # if self.with_logits:
        #     ypred = nn.Softmax(ypred, dim=-1)
        pred =  torch.clip(pred, self.epsilon, 1.0 - self.epsilon)
        if self.reduction == 'mean':
            return -torch.sum(torch.log(pred)*ref*c_weight, dim =-1 ).mean()
        elif self.reduction == 'sum':
            return -torch.sum(torch.log(pred)*ref*c_weight, dim =-1 ).sum()
        elif self.reduction is None:
            return -torch.sum(torch.log(pred)*ref*c_weight, dim =-1)
        else:
            print(f'The reduction {self.reduction} is not known')


class ContrastiveLoss(nn.Module):
   def __init__(self, batch_size, temperature=0.1):
       super().__init__()
       self.batch_size = batch_size
       self.temperature = temperature
       self.mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float()

   def calc_similarity_batch(self, a, b):
       representations = torch.cat([a, b], dim=0)
       return F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
   def put_on(self, a, b):
       return a.to(b.device)

   def forward(self, proj_1, proj_2):
       batch_size = proj_1.shape[0]
       z_i = F.normalize(proj_1, p=2, dim=-1)
       z_j = F.normalize(proj_2, p=2, dim=-1)

       similarity_matrix = self.calc_similarity_batch(z_i, z_j)

       sim_ij = torch.diag(similarity_matrix, batch_size)
       sim_ji = torch.diag(similarity_matrix, -batch_size)

       positives = torch.cat([sim_ij, sim_ji], dim=0)
       nominator = torch.exp(positives / self.temperature)

       denominator = self.put_on(self.mask, similarity_matrix) * torch.exp(similarity_matrix / self.temperature)

       all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
       loss = torch.sum(all_losses) / (2 * self.batch_size)
       return loss


def sampleSplitArray(root, phase='train', data='Sentinel', one_hot=False):
    sen_files = np.array(torch.load(f'{root}/{data}/{phase}/images/points.pt'))
    #sen_files = sen_files.reshape(sen_files.shape[0],sen_files.shape[1], 1) # because we are using single index
    label_files = np.array(torch.load(f'{root}/{data}/{phase}/labels/points.pt')).ravel()
    freq = np.unique(label_files, return_counts=True)
    weight = torch.tensor([v/sum(freq[1]) for v in freq[1]])

    if phase=='train':
        X_train, X_test, y_train, y_test = train_test_split(sen_files,
                label_files,
                test_size=0.2,
                train_size=0.8,
                random_state=10,
                shuffle=True,
                stratify=label_files)


        assert len(X_train) == len(y_train), 'Train images and labels not matching'
        assert len(X_test) == len(y_test), 'Test images and labels not matching'


        print('Validation sample done!')

        return (X_train, y_train), (X_test,y_test), weight
    else:
        return sen_files, label_files, weight

def normalize(tensor):
    return (tensor-tensor.min())/(tensor.max()-tensor.min())

def channel_swap(tensor):
    return tensor.permute(1, 0)


class PreDataset(Dataset):
    def __init__(self, snt, plt,tsx_ind, makenormal=True):
        tsx_ind = np.load(tsx_ind, allow_pickle=True).ravel()
        plnt = np.load(plt, allow_pickle=True)#[tsx_ind]
        snt = np.load(snt, allow_pickle=True)[tsx_ind]
        print('Array sape: ', tsx_ind.shape, snt.shape, plnt.shape)
        if makenormal:
            self.plnt = normalize(torch.from_numpy(plnt))
            self.snt = normalize(torch.from_numpy(snt))
        else:
            self.plnt = torch.from_numpy(plnt)
            self.snt = torch.from_numpy(snt)

        assert len(self.plnt) == len(snt), 'Length of dataset imagery and labells are not equal.'

    def __len__(self):
        return self.plnt.shape[0]

    def __getitem__(self, idx):
        s = self.snt[idx]
        p = self.plnt[idx]
        return s, p 


class PostDataset(Dataset):
    def __init__(self, snt, plt, lbl, tsx_ind, makenormal=True,onehot=True):
        tsx_ind = np.load(tsx_ind, allow_pickle=True).ravel()
        plnt = np.load(plt, allow_pickle=True)#[tsx_ind]
        snt = np.load(snt, allow_pickle=True)[tsx_ind]
        lbl = np.load(lbl, allow_pickle=True)[tsx_ind]
        #print(snt.shape, plnt.shape)
        if makenormal:
            self.plnt = normalize(torch.from_numpy(plnt))
            self.snt = normalize(torch.from_numpy(snt))
        else:
            self.plnt = torch.from_numpy(plnt)
            self.snt = torch.from_numpy(snt)

        if onehot:
            self.lbl = OneHot(lbl)
        else:
            self.lbl = torch.from_numpy(lbl)
        assert len(self.plnt) == len(snt), 'Length of dataset imagery and labells are not equal.'
        #print('Array shapes: ', self.snt.shape, self.plnt.shape, self.lbl.shape)

    def __len__(self):
        return self.plnt.shape[0]

    def __getitem__(self,idx):
        s = self.snt[idx]
        p = self.plnt[idx]
        y = self.lbl[idx]
        return s, p, y  # sentinel, planet, terrasar-x and labels

    
# class customLoader:
#     def __init__(self, root, batch_size=100, phase='train', data='Sentinel'):
#         self.root = root
#         self.phase = phase
#         self.batch_size = batch_size

#         #if self.phase == 'train':
#         self.train_files, self.valid_files, self.weight = sampleSplitArray(root=self.root,
#                                                                   phase='train',
#                                                                   data=data)
#         #if self.phase == 'test':
#         self.test_files = sampleSplitArray(root=self.root,
#                                                phase='test',
#                                                data=data)

#     def trainLoader(self):
#         train_dataset = SatDataset(self.train_files[0], self.train_files[1])
#         return DataLoader(dataset=train_dataset,batch_size = self.batch_size, drop_last=True, num_workers=0, shuffle=True)

#     def validLoader(self):
#         validation_dataset = SatDataset(self.valid_files[0], self.valid_files[1])
#         return DataLoader(dataset=validation_dataset, batch_size = self.batch_size, drop_last=True, num_workers=0, shuffle=True)

#     def testLoader(self):
#         test_dataset = SatDataset(self.test_files[0], self.test_files[1])
#         return DataLoader(dataset=test_dataset, batch_size = self.batch_size, drop_last=True, num_workers=0, shuffle=True)


class customLoader:
    def __init__(self, train_files=None, test_files=None, valid_files=None, batch_size=100, phase='train', data='sentinel'):
        self.batch_size = batch_size
        self.phase = phase
        self.train_files = train_files
        self.valid_files = valid_files
        self.weight = None #sampleSplitArray(root=self.root,phase='train',data=data)
        self.test_files = test_files #sampleSplitArray(root=self.root,phase='test',data=data)

    def trainLoader(self):
        if self.phase == 'pretrain':
            train_dataset = PreDataset(self.train_files[0], self.train_files[1],self.train_files[2])
        else:
            train_dataset = PostDataset(self.train_files[0], self.train_files[1],self.train_files[2],self.train_files[3])
        return DataLoader(dataset=train_dataset, batch_size = self.batch_size, drop_last=True, num_workers=0, shuffle=True)

    def validLoader(self):
        if self.valid_files is not None:
            validation_dataset = SatDataset(self.valid_files[0], self.valid_files[1])
            return DataLoader(dataset=validation_dataset, batch_size = self.batch_size, drop_last=True, num_workers=0, shuffle=True)
        else:
            return None

    def testLoader(self):
        if self.test_files is not None:
            test_dataset = PostDataset(self.test_files[0], self.test_files[1], self.test_files[2],self.test_files[3])
            return DataLoader(dataset=test_dataset, batch_size = self.batch_size, drop_last=False, num_workers=0, shuffle=False)
        else:
            return None


class TempCNN(torch.nn.Module):
    def __init__(self, input_dim=1, num_classes=11, sequencelength=45, kernel_size=3, hidden_dims=128, dropout=0.1):
        super(TempCNN, self).__init__()
        self.modelname = f"TempCNN_input-dim={input_dim[0]}_num-classes={num_classes}_sequencelenght={sequencelength[0]}_" \
                         f"kernelsize={kernel_size}_hidden-dims={hidden_dims}_dropout={dropout}"

        self.sequencelength = sequencelength
        self.hidden_dims = hidden_dims

        self.conv_bn_relu_a = Conv1D_BatchNorm_Relu_Dropout(input_dim[0], hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout)
        self.conv_bn_relu_b = Conv1D_BatchNorm_Relu_Dropout(input_dim[1], hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout)
        #self.conv_bn_relu_c = Conv1D_BatchNorm_Relu_Dropout(input_dim[2], hidden_dims, kernel_size=kernel_size,
        #                                                   drop_probability=dropout)

        self.conv_bn_relu2 = Conv1D_BatchNorm_Relu_Dropout(hidden_dims, hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout)
        self.conv_bn_relu3 = Conv1D_BatchNorm_Relu_Dropout(hidden_dims, hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout)
        self.flatten = Flatten()

        self.dense_a = FC_BatchNorm_Relu_Dropout(hidden_dims * sequencelength[0], 4 * hidden_dims, drop_probability=dropout)
        self.dense_b = FC_BatchNorm_Relu_Dropout(hidden_dims * sequencelength[1], 4 * hidden_dims, drop_probability=dropout)
        #self.dense_c = FC_BatchNorm_Relu_Dropout(hidden_dims * sequencelength[2], 4 * hidden_dims, drop_probability=dropout)
        self.fuser = nn.Linear(2*4 * hidden_dims, 4 * hidden_dims)
        self.softmax = nn.Sequential(nn.Linear(4 * hidden_dims, num_classes), nn.Softmax(dim=1))
        self.projector = nn.Sequential(
                nn.Linear(in_features=4*hidden_dims, out_features=hidden_dims),
                nn.BatchNorm1d(hidden_dims),
                nn.ReLU(),
                nn.Linear(in_features=hidden_dims, out_features=100),
                nn.BatchNorm1d(100))


    def forward(self, sn, pl, pre_train=True):
        sn = sn.transpose(1,2)
        sn = self.conv_bn_relu_a(sn)
        sn = self.conv_bn_relu2(sn)
        sn = self.conv_bn_relu3(sn)
        sn = self.flatten(sn)
        sn = self.dense_a(sn)


        pl = pl.transpose(1,2)
        pl = self.conv_bn_relu_b(pl)
        pl = self.conv_bn_relu2(pl)
        pl = self.conv_bn_relu3(pl)
        pl = self.flatten(pl)
        pl = self.dense_b(pl)

        if pre_train:
            sen_proj = self.projector(sn)
            plt_proj = self.projector(pl)
            return sen_proj, plt_proj
        else:
            fused = torch.cat((sn, pl), dim=-1) # fusion in linear 
            fused = feat = self.fuser(fused)
            fused = self.softmax(fused) # projects to crop probability classes
            return fused, feat

    def save(self, path="model.pth", **kwargs):
        print("\nsaving model to " + path)
        model_state = self.state_dict()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(dict(model_state=model_state, **kwargs), path)

    def load(self, path):
        print("loading model from " + path)
        snapshot = torch.load(path, map_location="cpu")
        model_state = snapshot.pop('model_state', snapshot)
        self.load_state_dict(model_state)
        return snapshot


class Conv1D_BatchNorm_Relu_Dropout(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, kernel_size=5, drop_probability=0.5):
        super(Conv1D_BatchNorm_Relu_Dropout, self).__init__()

        self.block = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dims, kernel_size, padding=(kernel_size // 2)),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Dropout(p=drop_probability)
        )

    def forward(self, X):
        return self.block(X)


class FC_BatchNorm_Relu_Dropout(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, drop_probability=0.5):
        super(FC_BatchNorm_Relu_Dropout, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Dropout(p=drop_probability)
        )

    def forward(self, X):
        return self.block(X)


class Flatten(nn.Module):
    def forward(self, input):
        val = input.view(input.size(0), -1)
        #print('Flatten shape: ', val.shape)
        return val


class Trainer:
    def __init__(self,
            pretrain_epochs=100,
            finetune_epochs=100,
            lr=0.0001,
            h_dims=10,
            n_layer=5,
            kernel=3,
            dropout=0.1,
            pre_train_loader=None,
            train_loader=None,
            valid_loader=None,
            test_loader=None,
            model_fold=None,
            rep_fold=None,
            data='sentinel',
            num_class=11,
            weight=None,
            index_fold=None,
            pred_scene_fold=None,
            feature_fold = None,
            scene_fold=None,
            raster_tempelate=None,
            model_name=None,
            temperature=None):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.pre_train_loader = pre_train_loader
        self.pretrain_epochs = pretrain_epochs
        self.finetune_epochs = finetune_epochs
        self.lr = lr
        self.h_dims = h_dims
        self.n_layer = n_layer
        self.kernel = kernel
        self.dropout = dropout
        self.weight = weight.to(self.device)
        self.rep_fold = f'{rep_fold}/{model_name}'
        self.model_fold = f'{model_fold}/{model_name}'
        self.model_name = model_name
        self.index_fold = index_fold
        self.pred_scene_fold = f'{pred_scene_fold}/{model_name}'
        self.feature_fold = f'{feature_fold}/{model_name}'
        self.scene_fold = scene_fold
        self.raster_tempelate = raster_tempelate
        self.data = data
        self.tloss = []
        self.vloss = []
        self.vacc = []
        self.tolerance = 10
        self.ac_monitor = 0
        self.petience = 10  # early stoping after 10 epochs

        if not os.path.exists(self.rep_fold):
            os.makedirs(self.rep_fold)
        if not os.path.exists(self.model_fold):
            os.makedirs(self.model_fold)
        if not os.path.exists(self.pred_scene_fold):
            os.makedirs(self.pred_scene_fold)
        if not os.path.exists(self.feature_fold):
            os.makedirs(self.feature_fold)


        if data == "sentinel":
            input_size = 4
            seq_len = 20
        elif data == 'terrasarx':
            input_size = 2
            seq_len = 7
        elif data == 'planet':
            input_size = 4
            seq_len = 11
        elif data == 'fused':
            input_size = 4
            seq_len = 31
        elif data == 'fused_deep':
            input_size = [4,4,2]
            seq_len = [20,11,7]
        elif data == 'fused_contrast':
            input_size = [4,2]
            seq_len = [11,7]
        else:
            raise ValeError(f'The specified dataset {data} not known')
        print(seq_len)
        print(f'Parameter summar \n learning rate: {self.lr}\n ndims : {self.h_dims}\n, nlayer: {self.n_layer}\n')
        self.model = TempCNN(input_dim=input_size,
                num_classes=num_class,
                sequencelength=seq_len,
                kernel_size=self.n_layer,
                hidden_dims=self.h_dims,
                dropout=self.dropout).to(self.device)
        
        self.optimizer_post = torch.optim.Adam(self.model.parameters(),lr=self.lr)
        #self.optimizer_pre = torch.optim.SGD(self.model.parameters(),lr=self.lr,momentum=0.9, weight_decay=0.00005)
        self.criterion_tune = CCE().to(self.device)
        self.criterion_pre =  ContrastiveLoss(batch_size=self.train_loader.batch_size, temperature=temperature).to(self.device)
        # self.best_weight = copy.deepcopy(self.model.state_dict())
        self.accuracy = 0
        self.f1 = 0
                         
    def pretrain(self):
        ac_monitor = 0
        for k in range(self.pretrain_epochs):
            epoch_loss = []
            for jj, (snt, plnt) in enumerate(self.pre_train_loader):
                snt = torch.permute(snt, (0,2,1)).to(self.device)
                plnt = torch.permute(plnt, (0,2,1)).to(self.device)
                #print(snt.shape, plnt.shape)
                #tsx = torch.permute(tsx, (0,2,1)).to(self.device)
                #y = y.to(self.device)
                
                #print(f'Shape of sen: {snt.shape}, shape of plnt: {plnt.shape}, tsx shape: {tsx.shape}, shape of y: {y.shape}')

                self.optimizer_post.zero_grad()
                sen_p,plt_p = self.model(snt.float(), plnt.float())
                #loss = F.nll_loss(logits, lbl.view(-1).long(), weight=self.weight.float())
                #loss = self.criterion(logits, y) #  weight=self.weight.float()
                loss = self.criterion_pre(sen_p, plt_p)
                epoch_loss.append(loss.item())
                loss.backward()
                self.optimizer_post.step()
                print(f'epoch {k} ->> step {jj} ->> loss: {loss.item()}', end = '\r', flush=True)

            self.tloss.append(np.nanmean(epoch_loss))

#             v_loss_e = []
#             vacc_e = []
                
#             for kk, (sentinel, lbl) in enumerate(self.valid_loader):
#                 sentinel = sentinel.to(self.device)
#                 lbl = lbl.to(self.device)

#                 vlogs = self.model(sentinel.float())
#                 vloss = F.nll_loss(vlogs, lbl.view(-1).long(), weight=self.weight.float())
#                 vac = accuracy_score(np.array(vlogs.argmax(dim=-1).cpu()).ravel(), np.array(lbl.view(-1).long().cpu()).ravel())

#                 v_loss_e.append(vloss.item())
#                 vacc_e.append(vac.item())

# self.vloss.append(np.nanmean(v_loss_e))
#             self.vacc.append(np.nanmean(vacc_e))

#             print(f'epoch {k} ->> vloss {np.nanmean(v_loss_e)} ->> acc: {np.nanmean(vacc_e)}')
                
#             if k >=1:
#                 if self.vloss[-1]<=max(self.vloss):
#                     self.best_weight = copy.deepcopy(self.model.state_dict())
#                     self.ac_monitor = 0
#                 else:
#                     self.ac_monitor+=1

            if k%5 == 0:
                model_path = f'{self.model_fold}/{self.data}_weight_pretrain_{self.lr}_{self.h_dims}_{self.n_layer}_tp.pt'
                torch.save(self.model.state_dict(), model_path)
            # if self.tolerance<self.ac_monitor:
            #     acc = f'{self.rep_fold}/{self.data}_vcc.pt'
            #     vls = f'{self.rep_fold}/{self.data}_vls.pt'
        tls = f'{self.rep_fold}/{self.data}_tloss_pretrain_{self.lr}_{self.h_dims}_{self.n_layer}_tp.pt'
        torch.save(torch.from_numpy(np.array(self.tloss)), tls)
                # torch.save(torch.from_numpy(np.array(self.vloss)), vls)
                # torch.save(torch.from_numpy(np.array(self.vacc)), acc)

        model_path = f'{self.model_fold}/{self.data}_pretrain_{self.lr}_{self.h_dims}_{self.n_layer}_tp.pt'
        torch.save(self.model.state_dict(), model_path)

        return None  # just for early stop and return non, can also made to return the model

    def finetune(self,read_weight=False,freez_weight=True, switch=50):
        #self.optimizer_post = torch.optim.Adam(self.model.parameters(),lr=self.lr)
        if read_weight:
            model_path = f'{self.model_fold}/{self.data}_weight_pretrain_{self.lr}_{self.h_dims}_{self.n_layer}_tp.pt'
            self.model.load_state_dict(torch.load(model_path))
            print('Weight loaded!')
            if freez_weight:
                for name, param in self.model.named_parameters():
                    if 'fuser' in name or 'softmax' in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
        ac_monitor = 0
        for k in range(self.finetune_epochs):
            epoch_loss = []
            for jj, (snt, plnt, y) in enumerate(self.train_loader):
                snt = torch.permute(snt, (0,2,1)).to(self.device)
                plnt = torch.permute(plnt, (0,2,1)).to(self.device)
                y = y.to(self.device)
                self.optimizer_post.zero_grad()
                logs,_ = self.model(snt.float(), plnt.float(), pre_train=False)
                #print(logs.shape, y.shape)
                loss = self.criterion_tune(logs,y,self.weight[:logs.shape[0]])
                epoch_loss.append(loss.item())
                loss.backward()
                self.optimizer_post.step()
                print(f'epoch {k} ->> step {jj} ->> loss: {loss.item()}', end = '\r', flush=True)
            if freez_weight:
                self.tloss.append(np.nanmean(epoch_loss))
                if k == switch:
                    for name, param in self.model.named_parameters():
                        param.requires_grad = True

            if k%5 == 0:
                model_path = f'{self.model_fold}/{self.data}_weight_tune_{self.lr}_{self.h_dims}_{self.n_layer}_tp.pt'
                torch.save(self.model.state_dict(), model_path)
        tls = f'{self.rep_fold}/{self.data}_tloss_tune_{self.lr}_{self.h_dims}_{self.n_layer}_tp.pt'
        torch.save(torch.from_numpy(np.array(self.tloss)), tls)

        model_path = f'{self.model_fold}/{self.data}_weight_tune_{self.lr}_{self.h_dims}_{self.n_layer}_tp.pt'
        torch.save(self.model.state_dict(), model_path)


    def test(self, read_weight=False, phase='optimize', save_feature=True):
        if read_weight:
            model_path = f'{self.model_fold}/{self.data}_weight_tune_{self.lr}_{self.h_dims}_{self.n_layer}_tp.pt'
            self.model.load_state_dict(torch.load(model_path))
        # else:
            # self.model.load_state_dict(self.best_weight)
            # self.model.eval()

        control = 0
        PREDS = []
        LBS = []
        print('Started Prediction')
        for jj, (snt, plnt, y) in enumerate(self.test_loader):
            snt = torch.permute(snt, (0,2,1)).to(self.device)
            plnt = torch.permute(plnt, (0,2,1)).to(self.device)
            #tsx = torch.permute(tsx, (0,2,1)).to(self.device)
            y = y.to(self.device)
                
            logs,feat_ = self.model(snt.float(), plnt.float(),pre_train=False)
            logs = logs.argmax(-1)
            ac = accuracy_score(np.array(y.argmax(-1).cpu()).ravel(), np.array(logs.cpu()).ravel())
            f1 = f1_score(np.array(y.argmax(-1).cpu()).ravel(), np.array(logs.cpu()).ravel(), average='weighted')
            self.accuracy+=ac
            self.f1+=f1
            control+=1
            PREDS.append(np.array(logs.cpu()).reshape(-1,1))
            LBS.append(np.array(y.argmax(-1).cpu()).reshape(-1,1))
            print(f'{jj}, Acc: {ac}, F1: {f1}')
            if save_feature:
                np.save(f'{self.feature_fold}/{jj}_{self.data}.npy', feat_.cpu().detach().numpy())
                np.save(f'{self.feature_fold}/{jj}_{self.data}_label.npy', y.cpu().detach().numpy())
        self.accuracy = self.accuracy/control
        self.f1 = self.f1/control
        # if phase != 'optimize':
        print('=======================================')
        print(f'Overall test acuracy: {self.accuracy}')
        print(f'Overall f-1 score: {self.f1}')
        print('=======================================')
        text = open(f'{self.rep_fold}/deep{self.data}_test_report_{self.lr}_{self.h_dims}_{self.n_layer}_tp.txt','a+')
        text.write(f"Overall accuracy: {self.accuracy}\n")
        text.write(f"Micro F-1 score: {self.f1}\n")

        PREDS = np.vstack(tuple(PREDS))
        LBS = np.vstack(tuple(LBS))
        cm_matrics = confusion_matrix(LBS.ravel(), PREDS.ravel(), normalize=None)
        names = ['Guizota','Maize','Millet','Others','Pepper','Teff']
        plot_confusion_matrix(
             cm=cm_matrics,
             title=f'{self.model_name}_deep_{self.data}_t',
             cmap=None,
             normalize=True,
             path=self.rep_fold,
             target_names=names,
             fname=f'{self.model_name}_deep_{self.data}_{self.lr}_{self.h_dims}_{self.n_layer}_tp',
             save=True)
        np.save(f'{self.rep_fold}/{self.model_name}_deep_{self.data}_test_{self.lr}_{self.h_dims}_{self.n_layer}_tp.npy',PREDS)

    def predictFullsene(self, save_raster=True, save_array=True):
        # self.model.load_state_dict(self.best_weight)
        self.model.eval()

        files = glob(f'{self.scene_fold}/{self.data}/*.npy')
        preds = []
        print('Scene prediction started')
        print(f'{len(files)} files obtained for prediction')
        for i in range(len(files)):
            file = normalize(torch.from_numpy(np.load(f'{self.scene_fold}/{self.data}/{i}.npy').astype(float))) # the same pipline with training
            #print('Shape of the array: ', file.shape)
            #file = normalize(torch.from_numpy(file.reshape(file.shape[0], file.shape[1],1)))
            pred = self.model(file.float().to(self.device)).argmax(-1)
            preds.append(np.array(pred.cpu()).reshape(-1,1))
            print(f'{(i/len(files))*100}% scene prediction done!',end='\r', flush=True)
        preds = np.vstack(tuple(preds))

        inds = np.load(f'{self.index_fold}/index_true.npy').reshape(-1,1)
        inds_all = np.load(f'{self.index_fold}/index_all.npy').astype(float).reshape(-1,1)

        final = np.empty(inds_all.shape, dtype=np.float)
        oks = (inds_all == 1).ravel().tolist()
        nok = (inds_all == 0).ravel().tolist()

        final[oks] = preds
        final[nok] = np.nan

        if save_raster:
            print('Writting prediction array')
            new_path = f'{self.pred_scene_fold}/{self.model_name}_{self.data}.npy'
            with open(new_path, 'wb') as ff:
                np.save(ff, final)

        if save_raster:
            print('writting to raster file...', end='\r', flush=True)
            rasterize(self.raster_tempelate, array=final, path=self.pred_scene_fold, name=f'{self.model_name}_{self.data}')

def argumentParser():
    parser = argparse.ArgumentParser(description='Runs and tests crop type mapping using Inception time')
    parser.add_argument('--data_root', help='Root folder that contained all tensors folders', type=str)
    parser.add_argument('--weight_path', help='path/folder to save weight', type = str)
    parser.add_argument('--log_path', help='Folder to save the reports and logs', type=str)
    parser.add_argument('--index_fold', help='Folder that contains no data value indexes', type=str)
    parser.add_argument('--pred_scene_fold', help='Folder to save full sceene predicted images', type=str)
    parser.add_argument('--scene_fold', help='Folder to save the reports and logs', type=str)
    parser.add_argument('--feature_fold', help='Folder to save the deep features', type=str)
    parser.add_argument('--raster_tempelate', help='Raster path to copy the temelate', type=str)
    parser.add_argument('--batch_size', help='batch size to load task datasets', default=100, type=int, required=False)
    parser.add_argument('--lr', help='learning rate for inner model', default=0.0001, type=float, required=False)
    parser.add_argument('--hdims', help='Hidden inner dimensions of the model', default=10, type=int, required=False)
    parser.add_argument('--nlayer', help='Model number of layers', default=3, type=int, required=False)
    parser.add_argument('--dropout', help='Percent layer dropout', default=0.01, type=float, required=False)
    parser.add_argument('--kernel', help='Kernel size for 1-d convolutions', default=3, type=int, required=False)
    parser.add_argument('--pretrain_epochs', help='number of epochs to pre-train', default=100, type=int, required=False)
    parser.add_argument('--finetune_epochs', help='number of epochs to finetune', default=100, type=int, required=False)
    parser.add_argument('--runner', help='Whether train or test', default='train_test', type=str, required=False)
    parser.add_argument('--data', help='Dataset to train', default='sentinel', type=str, required=False)
    parser.add_argument('--model', help='Model name for optimization and test', default='LSTM', type=str, required=False)
    parser.add_argument('--temperature', help='Energy parameter for cooling temperature in the infoNCE loss', default=0.1, type=float, required=False)
    args = parser.parse_args()
    return args
    

if __name__ == "__main__":
    args = argumentParser()
    if args.runner == 'train_test':
        sen_tr = f'/home/getch/ssl/EO_Africa/samples/sentinel/train_test/x_train.npy'
        plnt_tr = f'/home/getch/ssl/EO_Africa/samples/planet/train_test/x_train.npy'
        tsx_tr = f'/home/getch/ssl/EO_Africa/samples/terrasarx/train_test/x_train.npy'
        ytrain = f'/home/getch/ssl/EO_Africa/samples/sentinel/train_test/y_train.npy'
        vals = np.load(ytrain)
        uniqs = np.unique(vals, return_counts=True)
        print('Uniqs list: ', uniqs)
        total = np.sum(uniqs[1])
        c_weight = [1-uniqs[0][i]/total for i in range(len(uniqs[0]))]
        weight = torch.from_numpy(np.array([c_weight]*total))
        tr_ind = f'/home/getch/ssl/EO_Africa/samples/train_index.npy'
        
        sen_ts = f'/home/getch/ssl/EO_Africa/samples/sentinel/train_test/x_test.npy'
        plnt_ts = f'/home/getch/ssl/EO_Africa/samples/planet/train_test/x_test.npy'
        tsx_ts = f'/home/getch/ssl/EO_Africa/samples/terrasarx/train_test/x_test.npy'
        ytest = f'/home/getch/ssl/EO_Africa/samples/sentinel/train_test/y_test.npy'
        ts_ind = f'/home/getch/ssl/EO_Africa/samples/test_index.npy'
        
        sen_ptr = '/home/getch/ssl/EO_Africa/FILL/sentinel_feat.npy'
        plnt_ptr = '/home/getch/ssl/EO_Africa/FILL/planet_feat.npy'
        tsx_ptr = '/home/getch/ssl/EO_Africa/FILL/terrasarx_feat.npy'
        tsx_ptr_ind = '/home/getch/ssl/EO_Africa/FILL/mask.npy'

        ptr_loader = customLoader(phase='pretrain',
            train_files=[plnt_ptr, tsx_ptr, tsx_ptr_ind],
            test_files=None,
            valid_files=None,
            batch_size=args.batch_size,
            data=args.data)

        fnt_loader = customLoader(phase='finetune',
                train_files=[plnt_tr, tsx_tr,ytrain,tr_ind],
                test_files=[plnt_ts, tsx_ts, ytest, ts_ind],
                valid_files=None,
                batch_size=args.batch_size,
                data=args.data)
        #loader = customLoader(
        #    train_files=[sen_tr, plnt_tr, tsx_tr, ytrain, tr_ind],
        #    test_files=[sen_ts, plnt_ts, tsx_ts,ytest, ts_ind],
        #    valid_files=None,
        #    batch_size=args.batch_size,
        #    data=args.data)

        trainer = Trainer(pretrain_epochs=args.pretrain_epochs,
                finetune_epochs=args.finetune_epochs,
                lr=args.lr,
                h_dims=args.hdims,
                n_layer=args.nlayer,
                dropout=args.dropout,
                kernel=args.kernel,
                pre_train_loader=ptr_loader.trainLoader(),
                train_loader=fnt_loader.trainLoader(),
                valid_loader=fnt_loader.validLoader(),
                test_loader=fnt_loader.testLoader(),
                model_fold=args.weight_path,
                rep_fold=args.log_path,
                data=args.data,
                num_class=6,
                index_fold=args.index_fold,
                pred_scene_fold=args.pred_scene_fold,
                scene_fold=args.scene_fold,
                feature_fold=args.feature_fold,
                raster_tempelate=args.raster_tempelate,
                model_name=args.model,
                temperature=args.temperature,
                weight=weight)

        #trainer.pretrain()
        trainer.finetune(read_weight=True,freez_weight=False)
        trainer.test()
        # trainer.predictFullsene()

#     elif args.runner == 'test':
#         loader = customLoader(root=args.data_root,
#                      f         batch_size=args.batch_size,
#                               phase='test',
#                               data=args.data)
#         tester = Tester(test_loader=loader.testLoader(),
#                         model_fold=args.weight_path,
#                         rep_fold=args.log_path,
#                         data=args.data)
#         tester.test()
        
#     elif args.runner == 'optimize':
#         loader= customLoader(root=args.data_root,
#                                 batch_size=args.batch_size,
#                                 phase='train',
#                                 data = args.data)

        
#         lrs = [0.01, 0.001,0.0001,0.00001]
#         hdims = [2 **2, 2**3, 2**5, 2**6, 2**7, 2**8]
#         kernels = [3, 5]
#         drops = [0, 0.1, 0.2, 0.3]

#         text = open(f'{args.log_path}/{args.model}/{args.data}_{args.model}_optimize.txt','w')
#         c = 0
#         tot = len(lrs)*len(hdims)*len(kernels)
#         for lr in lrs:
#             for hdim in hdims:
#                 for kernel in kernels:
#                     for drop in drops:
#                         print(f'learning rate:{lr} \n number of huuden units: {hdim}\n number of kernels: {kernel}\n, Percent done: {((c+1)/tot)*100}\n')
#                         OPTIMIZER = Trainer(epochs=args.epochs,
#                                 lr=lr,
#                                 h_dims=hdim,
#                                 n_layer=args.nlayer,
#                                 dropout=drop,
#                                 kernel=kernel,
#                                 train_loader=loader.trainLoader(),
#                                 valid_loader=loader.validLoader(),
#                                 test_loader=loader.testLoader(),
#                                 model_fold=args.weight_path,
#                                 rep_fold=args.log_path,
#                                 data=args.data,
#                                 num_class=11,
#                                 index_fold=args.index_fold,
#                                 pred_scene_fold=args.pred_scene_fold,
#                                 scene_fold=args.scene_fold,
#                                 raster_tempelate=args.raster_tempelate,
#                                 model_name=args.model,
#                                 weight=loader.weight)
#                         OPTIMIZER.train()
#                         OPTIMIZER.test(phase=args.runner)
#                         text.write(f'accuracy: {OPTIMIZER.accuracy}, lr: {lr}, hdim: {hdim}, kernel: {kernel}, dropout:{drop}\n')
#                         c+=1
#         text.close()


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import os
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator
import argparse
import itertools
from torch import nn
from torch.nn import functional as F
from torchsampler import ImbalancedDatasetSampler
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
        
    def forward(self, pred, ref):
        # if pred.shape != ref.shape:
        #     raise ValueError('The categorical cross entropy is expecting arrays of equal size')
        # if len(ref.shape) == 1:
        #     print('The CCE loss is expecing the predicted values as one-hot encoded vectors, changes will be applied')
        #     ypred = self.onehot(ypred)
        # if self.with_logits:
        #     ypred = nn.Softmax(ypred, dim=-1)
        pred =  torch.clip(pred, min=self.epsilon, max=(1.0 - self.epsilon))
        if self.reduction == 'mean':
            return -torch.sum(torch.log(pred)*ref, dim =-1).mean()
        elif self.reduction == 'sum':
            return -torch.sum(torch.log(pred)*ref, dim =-1).sum()
        elif self.reduction is None:
            return -torch.sum(torch.log(pred)*ref, dim =-1)
        else:
            print(f'The reduction {self.reduction} is not known')

def sampleSplitArray(root, phase='train', data='Sentinel', one_hot=False):
    sen_files = np.array(torch.load(f'{root}/{data}/{phase}/images/points.pt'))
    sen_files = sen_files.reshape(sen_files.shape[0],sen_files.shape[1], 1) # because we are using single index
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



class SatDataset(Dataset):
    def __init__(self, plt, lbl, makenormal=True,onehot=True):
        plt = np.load(plt, allow_pickle=True)
        lbl = np.load(lbl, allow_pickle=True)
        
        
        if makenormal:
            plt = normalize(plt)
            self.plt = torch.from_numpy(plt)
        else:
            self.plt = torch.from_numpy(plt)
            
        if onehot:
            self.lbl = OneHot(lbl)
        else:
            self.lbl = torch.from_numpy(lbl)

        assert len(self.plt) == len(self.lbl), 'Length of dataset imagery and labells are not equal.'

    def __len__(self):
        return len(self.plt)

    def __getitem__(self, idx):
        x = self.plt[idx]
        y = self.lbl[idx]
        return x, y
    
    

# class customLoader:
#     def __init__(self, root, batch_size=100, phase='train', data='Sentinel'):
#         self.root = root
#         self.phase = phase
#         self.batch_size = batch_size

#         if self.phase == 'train':
#             self.train_files, self.valid_files, self.weight = sampleSplitArray(root=self.root,
#                                                                   phase='train',
#                                                                   data=data)
#         if self.phase == 'test':
#             self.test_files = sampleSplitArray(root=self.root,
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
    def __init__(self, train_files=None, test_files=None, valid_files=None, batch_size=100, data='sentinel'):
        # self.root = root
        # self.phase = phase
        self.batch_size = batch_size

        #if self.phase == 'train':
        self.train_files = train_files
        self.valid_files = valid_files
        self.weight = None #sampleSplitArray(root=self.root,phase='train',data=data)
        #if self.phase == 'test':
        self.test_files = test_files #sampleSplitArray(root=self.root,phase='test',data=data)

    def trainLoader(self):
        train_dataset = SatDataset(self.train_files[0], self.train_files[1])
        return DataLoader(dataset=train_dataset, batch_size = self.batch_size, drop_last=False, num_workers=0, shuffle=True)

    def validLoader(self):
        if self.valid_files is not None:
            validation_dataset = SatDataset(self.valid_files[0], self.valid_files[1])
            return DataLoader(dataset=validation_dataset, batch_size = self.batch_size, drop_last=False, num_workers=0, shuffle=True)
        else:
            return None

    def testLoader(self):
        test_dataset = SatDataset(self.test_files[0], self.test_files[1])
        return DataLoader(dataset=test_dataset, batch_size = self.batch_size, drop_last=False, num_workers=0, shuffle=False)


class ClassificationModel(ABC,torch.nn.Module, BaseEstimator):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self):
        pass # return logprobabilities

    @torch.no_grad()
    def predict(self, logprobabilities):
        return  logprobabilities.argmax(-1)

    @abstractmethod
    def save(self, path="model.pth",**kwargs):
        pass

    @abstractmethod
    def load(self, path):
        pass #return snapshot


class RNN(ClassificationModel):
    def __init__(self, input_dim=1, hidden_dims=3, nclasses=11, num_rnn_layers=1, dropout=0.2, bidirectional=False,use_batchnorm=False, use_attention=False, use_layernorm=True):

        super(RNN, self).__init__()

        self.nclasses=nclasses
        self.use_batchnorm = use_batchnorm
        self.use_attention = use_attention
        self.use_layernorm = use_layernorm
        self.bidirectional = bidirectional

        self.d_model = num_rnn_layers*hidden_dims

        if use_layernorm:
            self.inlayernorm = nn.LayerNorm(input_dim)
            self.clayernorm = nn.LayerNorm((hidden_dims + hidden_dims*bidirectional)*num_rnn_layers)

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dims, num_layers=num_rnn_layers,
                            bias=False, batch_first=True, dropout=dropout, bidirectional=bidirectional)

        if bidirectional: 
            hidden_dims = hidden_dims * 2

        outlineardims = hidden_dims if use_attention else hidden_dims*num_rnn_layers
        self.linear_class = nn.Linear(outlineardims, nclasses, bias=True)

        if use_batchnorm:
            self.bn = nn.BatchNorm1d(hidden_dims)


    def _logits(self, x):

        if self.use_layernorm:
            x = self.inlayernorm(x)

        outputs, last_state_list = self.lstm(x)

        h, c = last_state_list
        if self.use_attention:
            if self.bidirectional:
                query_forward = c[-1]
                query_backward = c[-2]
                query = torch.cat([query_forward, query_backward],1)
            else:
                query = c[-1]

            #h, weights = self.attention(query.unsqueeze(1), outputs)
            h = h.squeeze(1)
        else:
            nlayers, batchsize, n_hidden = c.shape
            h = self.clayernorm(c.transpose(0,1).contiguous().view(batchsize,nlayers*n_hidden))

        logits = self.linear_class(h)

        if self.use_attention:
            pts = weights
        else:
            pts = None

        return logits, None, pts, None

    def forward(self,x):
        logits, deltas, pts, budget = self._logits(x)

        probabilities = nn.functional.softmax(logits, dim=1)
        # stack the lists to new tensor (b,d,t,h,w)
        return probabilities, deltas, pts, budget

    def save(self, path="model.pth", **kwargs):
        print("\nsaving model to "+path)
        model_state = self.state_dict()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(dict(model_state=model_state,**kwargs),path)

    def load(self, path):
        print("loading model from "+path)
        snapshot = torch.load(path, map_location="cpu")
        model_state = snapshot.pop('model_state', snapshot)
        self.load_state_dict(model_state)
        return snapshot

    
class Trainer:
    def __init__(self,epochs=100,
                 lr=0.0001,
                 h_dims=10,
                 n_layer=5,
                 p_drops=0.2,
                 train_loader=None,
                 valid_loader=None,
                 test_loader=None,
                 model_fold=None,
                 rep_fold=None,
                 pred_scene_fold=None,
                 data='sentinel',
                 num_class=11,
                 model_name=None,
                 weight=None):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        #self.acc_fn = torchmetrics.Accuracy(task="multiclass", num_classes=6).to(self.device)
        self.epochs = epochs
        self.lr = lr
        self.h_dims = h_dims
        self.n_layer = n_layer
        self.p_drops = p_drops
        self.data = data
        self.weight = weight # .cuda()
        self.rep_fold = f'{rep_fold}/{model_name}'
        self.model_fold = f'{model_fold}/{model_name}'
        self.pred_scene_fold = f'{pred_scene_fold}/{model_name}'
        self.model_name = model_name


        if not os.path.exists(self.rep_fold):
            os.makedirs(self.rep_fold)
        if not os.path.exists(self.model_fold):
            os.makedirs(self.model_fold)
        if not os.path.exists(self.pred_scene_fold):
            os.makedirs(self.pred_scene_fold )



        self.tloss = []
        self.vloss = []
        self.vacc = []

        self.accuracy = 0
        self.f1 = 0

        self.tolerance = 10
        self.ac_monitor = 0
        self.petience = 10  # early stoping after 10 epochs
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
        else:
            raise ValeError(f'The specified dataset {data} not known')
        print(f'Parameter summar \n learning rate: {self.lr}\n ndims : {self.h_dims}\n, nlayer: {self.n_layer}\n')
        
        self.model = RNN(input_dim=input_size,
                         hidden_dims=self.h_dims,
                         nclasses=num_class,
                         num_rnn_layers=self.n_layer,
                         dropout=self.p_drops,
                         bidirectional=False,
                         use_batchnorm=True,
                         use_attention=False,
                         use_layernorm=True)
        
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.lr) ## betas=(0.9, 0.999), eps=1e-08
        # self.best_weight = copy.deepcopy(self.model.state_dict())
        self.criterion = CCE().to(self.device)
        self.accuracy = 0
                         
    def train(self):
        ac_monitor = 0
        for k in range(self.epochs):
            epoch_loss = []
            for jj, (x, y) in enumerate(self.train_loader):
                x = torch.permute(x, (0,2,1)).to(self.device)
                y = y.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(x.float())[0]
                loss = self.criterion(logits, y)
                # loss = F.nll_loss(logits, y, weight=self.weight.float())
                epoch_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()
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
#             self.vloss.append(np.nanmean(v_loss_e))
#             self.vacc.append(np.nanmean(vacc_e))

#             print(f'epoch {k} ->> vloss {np.nanmean(v_loss_e)} ->> acc: {np.nanmean(vacc_e)}')
                
#             if k >=1:
#                 if self.vloss[-1]<self.vloss[-2]:
#                     self.best_weight = copy.deepcopy(self.model.state_dict())
#                     self.ac_monitor = 0
#                 else:
#                     self.ac_monitor+=1

            if k%5 == 0:
                model_path = f'{self.model_fold}/weight.pt'
                torch.save(self.model.state_dict(), model_path)
            # if self.tolerance<self.ac_monitor:
                #acc = f'{self.rep_fold}/vcc.pt'
                # vls = f'{self.rep_fold}/vls.pt'
        tls = f'{self.rep_fold}/tloss.pt'
        torch.save(torch.from_numpy(np.array(self.tloss)), tls)
                # torch.save(torch.from_numpy(np.array(self.vloss)), vls)
                #torch.save(torch.from_numpy(np.array(self.vacc)), acc)

        model_path = f'{self.model_fold}/weight.pt'
        torch.save(self.model.state_dict(), model_path)

        return None  # just for early stop and return non, can also made to return the model
    def test(self, read_weight=False, phase='optimize'):
        if read_weight:
            model_path = f'{self.model_fold}/{self.data}_weight.pt'
            self.model.load_state_dict(torch.load(model_path))
        # else:
            # self.model.load_state_dict(self.best_weight)
            # self.model.eval()

        control = 0
        PREDS = []
        LBS = []
        print('Started Prediction')
        for j, (x, y) in enumerate(self.test_loader):
            print(j)
            x = torch.permute(x, (0,2,1)).to(self.device)
            y = y.to(self.device)
            logs = self.model(x.float())[0].argmax(-1)
            ac = accuracy_score(np.array(y.argmax(-1).cpu()).ravel(), np.array(logs.cpu()).ravel())
            f1 = f1_score(np.array(y.argmax(-1).cpu()).ravel(), np.array(logs.cpu()).ravel(), average='weighted')
            self.accuracy+=ac
            self.f1+=f1
            control+=1
            PREDS.append(np.array(logs.cpu()).reshape(-1,1))
            LBS.append(np.array(y.argmax(-1).cpu()).reshape(-1,1))
            print(f'{j}, Acc: {ac}, F1: {f1}')
        self.accuracy = self.accuracy/control
        self.f1 = self.f1/control        
        print('=======================================')
        print(f'Overall test acuracy: {self.accuracy}')
        print(f'Overall accuracy f-1: {self.f1}')
        print('=======================================')
        text = open(f'{self.rep_fold}/{self.data}_test_report.txt','w')
        text.write(f"Overall accuracy: {self.accuracy}\n")
        text.write(f"Micro F-1 score: {self.f1}\n")

        PREDS = np.vstack(tuple(PREDS))
        LBS = np.vstack(tuple(LBS))
        cm_matrics = confusion_matrix(LBS.ravel(), PREDS.ravel(), normalize=None)
        names = ['Guizota','Maize','Millet','Others','Pepper','Teff']
        plot_confusion_matrix(
             cm=cm_matrics,
             title=f'{self.model_name}_{self.data}',
             cmap=None,
             normalize=True,
             path=self.rep_fold,
             target_names=names,
             fname=f'{self.model_name}_{self.data}',
             save=True)
        np.save(f'{self.rep_fold}/{self.model_name}_{self.data}_test.npy',PREDS)
    
def argumentParser():
    parser = argparse.ArgumentParser(description='Runs and tests crop type mapping using Inception time')
    parser.add_argument('--data_root', help='Root folder that contained all tensors folders', type=str)
    parser.add_argument('--weight_path', help='path/folder to save weight', type = str)
    parser.add_argument('--log_path', help='Folder to save the reports and logs', type=str)
    parser.add_argument('--index_fold', help='Folder that contains no data value indexes', type=str)
    parser.add_argument('--pred_scene_fold', help='Folder to save full sceene predicted images', type=str)
    parser.add_argument('--scene_fold', help='Folder to save the reports and logs', type=str)
    parser.add_argument('--raster_tempelate', help='Raster path to copy the temelate', type=str)
    parser.add_argument('--batch_size', help='batch size to load task datasets', default=100, type=int, required=False)
    parser.add_argument('--lr', help='learning rate for inner model', default=0.0001, type=float, required=False)
    parser.add_argument('--hdims', help='Hidden inner dimensions of the model', default=10, type=int, required=False)
    parser.add_argument('--nlayer', help='Model number of layers', default=3, type=int, required=False)
    parser.add_argument('--dropout', help='Percent layer dropout', default=0.01, type=float, required=False)
    parser.add_argument('--kernel', help='Kernel size for 1-d convolutions', default=3, type=int, required=False)
    parser.add_argument('--epochs', help='number of epochs to train', default=100, type=int, required=False)
    parser.add_argument('--runner', help='Whether train or test', default='train_test', type=str, required=False)
    parser.add_argument('--data', help='Dataset to train', default='sentinel', type=str, required=False)
    parser.add_argument('--model', help='Model name for optimization and test', default='LSTM', type=str, required=False)
    args = parser.parse_args()
    return args
    

if __name__ == "__main__":
    args = argumentParser()
    if args.runner == 'train_test':
        xtrain = f'/home/getch/ssl/EO_Africa/samples/{args.data}/train_test/x_train.npy'
        ytrain = f'/home/getch/ssl/EO_Africa/samples/{args.data}/train_test/y_train.npy'
        xtest = f'/home/getch/ssl/EO_Africa/samples/{args.data}/train_test/x_test.npy'
        ytest = f'/home/getch/ssl/EO_Africa/samples/{args.data}/train_test/y_test.npy'
        loader = customLoader(
            train_files=[xtrain, ytrain],
            test_files=[xtest,ytest],
            valid_files=None,
            batch_size=args.batch_size,
            data=args.data)
        
        trainer = Trainer(epochs=args.epochs,
                          lr=args.lr,
                          h_dims=args.hdims,
                          n_layer=args.nlayer,
                          p_drops=args.dropout,
                          train_loader=loader.trainLoader(),
                          valid_loader=loader.validLoader(),
                          test_loader=loader.testLoader(),
                          model_fold=args.weight_path,
                          rep_fold=args.log_path,
                          data=args.data,
                          num_class=6,
                          model_name=args.model,
                          weight=None)
    

        trainer.train()
        trainer.test()
        # trainer.predictFullsene()

#     elif args.runner == 'test':
#         loader = customLoader(root=args.data_root,
#                               batch_size=args.batch_size,
#                               phase='test',
#                               data=args.data)
#         tester = Tester(test_loader=loader.testLoader(),
#                         model_fold=args.weight_path,
#                         rep_fold=args.log_path,
#                         data=args.data)
#         tester.test()
        
#     elif args.runner == 'optimize':
#         loader_a = customLoader(root=args.data_root,
#                                 batch_size=args.batch_size,
#                                 phase='train',
#                                 data = args.data)
#         loader_b = customLoader(root=args.data_root,
#                                 batch_size=args.batch_size,
#                                 phase='test',
#                                 data=args.data)
#         lrs = [0.0001, 0.00001]
#         hdims = [2 **2, 2**2, 2**5, 2**6, 2**7, 2**8]
#         nlayers = [1, 2, 3, 4, 5, 6, 7]
#         drops = [0, 0.1, 0.2, 0.3, 0.4]
        
#         text = open(f'{args.log_path}/report.txt', 'w')
        
#         for lr in lrs:
#             for hdim in hdims:
#                 for nlayer in nlayers:
#                     for drop in drops:
#                         if nlayer == 1:
#                             drop = 0
#                         print(f'learning rate:{lr} \n number of huuden units: {hdim}\n number of layers: {nlayer}\n dropout: {drop}')
#                         OPTIMIZER = Trainer(epochs=args.epochs,
#                                             lr=lr,
#                                             h_dims=hdim,
#                                             n_layers=nlayer,
#                                             p_drops=drop,
#                                             train_loader=loader_a.trainLoader(),
#                                             valid_loader=loader_a.validLoader(),
#                                             test_loader=loader_b.testLoader(),
#                                             model_fold=args.weight_path,
#                                             rep_fold=args.log_path,
#                                             data=args.data,
#                                             num_class=11,
#                                             weight=loader_a.weight)
#                         OPTIMIZER.train()
#                         OPTIMIZER.test()
#                         text.write(f'accuracy: {OPTIMIZER.accuracy}, lr: {lr}, hdim: {hdim}, nlayer: {nlayer}, dropout: {drop} \n')
#         text.close()

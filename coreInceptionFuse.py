import argparse
import itertools
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np 
from torch.utils.data import Dataset
from torchmetrics import Accuracy
from torch.utils.data import DataLoader
import copy
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score
from InceptionFuseNet import InceptionFuseNet   # this is mainly for deep learning
from utils import *

def sampleSplit(root, index_file, phase='train'):
    
    indexes = sorted(torch.load(index_file).tolist()) # list of indexes
    tsx_files = sorted(glob(f'{root}/TSX/{phase}/images/*.pt'))
    sen_files = sorted(glob(f'{root}/Sentinel/{phase}/images/*.pt'))
    plan_files = sorted(glob(f'{root}/Planet/{phase}/images/*.pt'))
    label_files = sorted(glob(f'{root}/TSX/{phase}/labels/*.pt'))
    
    tsx_sample = [tsx_files[ind] for ind in indexes]
    sen_sample = [sen_files[ind] for ind in indexes]
    plan_sample = [plan_files[ind] for ind in indexes]
    label_sample = [label_files[ind] for ind in indexes]
    print('Spliting done!')
    assert len(tsx_sample) == len(sen_sample) == len(plan_sample) == len(label_sample), 'Validation images and labels not matching'
    
    return sen_sample, plan_sample,tsx_sample, label_sample

def generateIndex(label_file, ratio=0.1):
    ys = np.array(torch.load(label_file).tolist())
    inds = np.array(range(len(ys))).reshape(-1,1)
    assert inds.shape == ys.shape, 'index and labels shape dod not much'
    X_train, X_valid, y_train, y_valid = train_test_split(inds, ys, test_size=ratio, random_state=2, stratify=ys)

    return X_train, X_valid

def sampleSplitArray(root, phase='train', one_hot=False):
    
    tsx_files = torch.load(f'{root}/TSX/{phase}/images/points.pt')
    sen_files = torch.load(f'{root}/Sentinel/{phase}/images/points.pt')
    plan_files = torch.load(f'{root}/Planet/{phase}/images/points.pt')
    label_files = torch.load(f'{root}/TSX/{phase}/labels/points.pt')

    if torch.isnan(plan_files).any():
        print(f'Planet has nan values')
        plan_files = torch.nan_to_num(plan_files, nan=0.0)
    if torch.isnan(sen_files).any():
        print(f'Sentinel has nan values')
        sen_files = torch.nan_to_num(sen_files, nan=0.0)
    if torch.isnan(tsx_files).any():
        print(f'TerraSAR-X has nan values')
        tsx_files = torch.nan_to_num(tsx_files, nan=0.0)
    if torch.isnan(label_files).any():
        print(f'Label has nan values')

    freq = np.unique(label_files, return_counts=True)
    weight = torch.tensor([v/sum(freq[1]) for v in freq[1]])
    
    if phase=='train':
        #if t_index is None or v_index is None:
        ind_t, ind_v = generateIndex(f'{root}/TSX/{phase}/labels/points.pt')
        
         #   ind_t = torch.load(t_index).tolist() # list of indexes
          #  ind_v = torch.load(v_index).tolist()
        print('Index files loaded')
    
        tsx_sample_t = tsx_files[ind_t]
        sen_sample_t = sen_files[ind_t] 
        plan_sample_t = plan_files[ind_t]
        label_sample_t = label_files[ind_t]
        if one_hot:
            label_sample_t = torch.nn.functional.one_hot(label_sample_t.long(), num_classes=11).squeeze()

        assert len(tsx_sample_t) == len(sen_sample_t) == len(plan_sample_t) == len(label_sample_t), 'Test images and labels not matching'

        tsx_sample_v = tsx_files[ind_v]
        sen_sample_v = sen_files[ind_v] 
        plan_sample_v = plan_files[ind_v]
        label_sample_v = label_files[ind_v]

        assert len(tsx_sample_v) == len(sen_sample_v) == len(plan_sample_v) == len(label_sample_v), 'Test images and labels not matching'

        print('Validation sample done!')

        return (sen_sample_t, plan_sample_t,tsx_sample_t, label_sample_t), (sen_sample_v, plan_sample_v,tsx_sample_v, label_sample_v), weight
    else:
        if one_hot:
            label_files = torch.nn.functional.one_hot(label_files.long(),num_classes=11).squeeze()
        return sen_files, plan_files, tsx_files, label_files
    
def normalize(tensor):
    return (tensor-tensor.min())/(tensor.max()-tensor.min())

def channel_swap(tensor):
    return tensor.permute(1, 0)


class SatDataset(Dataset):
    def __init__(self, snt, plt, tsx, lbl, read=False, normalize=True):
        self.plt = plt
        self.snt = snt
        self.tsx = tsx
        self.lbl = lbl
        self.read = read
        self.normalize = normalize
        
        assert len(self.plt) == len(self.snt) == len(self.tsx) == len(self.lbl), 'Length of dataset imagery and labells are not equal.'
        
    def __len__(self):
        return len(self.plt)
    
    def __getitem__(self, idx):
        planet = self.plt[idx]
        sentinel = self.snt[idx]
        terrasar = self.tsx[idx]
        label = self.lbl[idx]
        
        if self.read:
            snx = torch.load(sentinel).permute(1,0)
            plx = torch.load(planet).permute(1,0)
            trx = torch.load(terrasar).permute(1,0)
            lby = torch.load(label)
            if self.normalize:
                return normalize(snx), normalize(plx), normalize(trx), lby
            else:
                return snx, plx, trx, lby
        else:
            if self.normalize:
                return normalize(sentinel), normalize(planet), normalize(terrasar), label
            else:
                return sentinel, planet, terrasar, label

class customLoader:
    def __init__(self, root, index_root, batch_size=100, phase='train'):
        self.root = root
        self.phase = phase
        self.index_root = index_root
        self.batch_size = batch_size
        self.train_index = f'{index_root}/train_index.pt'
        self.valid_index = f'{index_root}/valid_index.pt'
    
        self.train_files, self.valid_files, self.weight = sampleSplitArray(root=self.root, phase='train')
        self.test_files = sampleSplitArray(root=self.root, phase='test')
                                            

    def trainLoader(self):
        trainset = SatDataset(self.train_files[0], self.train_files[1], self.train_files[2], self.train_files[3]) 
        return DataLoader(dataset=trainset, batch_size = self.batch_size, drop_last=True, num_workers=0, shuffle=True)
    
    def validLoader(self):
        validset = SatDataset(self.valid_files[0], self.valid_files[1], self.valid_files[2], self.valid_files[3])
        return DataLoader(dataset=validset, batch_size = self.batch_size, drop_last=True, num_workers=0, shuffle=True)
    
    def testLoader(self):
        testset = SatDataset(self.test_files[0], self.test_files[1], self.test_files[2], self.test_files[3])
        return DataLoader(dataset=testset, batch_size = self.batch_size, drop_last=True, num_workers=0, shuffle=True)
    

class Trainer:
    def __init__(self,
                 epochs=100,
                 lr=0.0001,
                 train_loader=None,
                 valid_loader=None,
                 test_loader=None,
                 class_weight=None,
                 num_layer=None,
                 h_dim=None,
                 use_bias=None,
                 with_pool=None,
                 model_name=None,
                 model_fold=None,
                 rep_fold=None,
                 index_fold=None,
                 pred_scene_fold=None,
                 scene_fold=None,
                 raster_tempelate=None):
        
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.class_weight = class_weight
        self.acc_fn = Accuracy(task="multiclass", num_classes=11)
        self.epochs = epochs
        self.lr = lr
        self.num_layer=num_layer
        self.h_dim=h_dim
        self.use_bias=use_bias
        self.with_pool=with_pool
        self.model_name=model_name
        self.rep_fold = rep_fold
        self.model_fold = model_fold
        self.index_fold=index_fold
        self.pred_scene_fold=pred_scene_fold
        self.scene_fold=scene_fold
        self.raster_tempelate=raster_tempelate

        if not os.path.exists(rep_fold):
            os.makedirs(rep_fold, exist_ok=True)
        if not os.path.exists(pred_scene_fold):
            os.makedirs(pred_scene_fold, exist_ok=True)
        if not os.path.exists(model_fold):
            os.makedirs(model_fold, exist_ok=True)

        self.tloss = []
        self.vloss = []
        self.vacc = []
        self.accuracy = 0
        self.f1 = 0
        
        self.petience = 10  
        self.device = torch.device('cpu')
        self.model = InceptionFuseNet(num_classes=11,
                                      input_dim=[4,4,2],
                                      num_layers=self.num_layer,
                                      hidden_dims=self.h_dim,
                                      use_bias=self.use_bias,
                                      use_residual= True,
                                      device=self.device,
                                      with_pool=self.with_pool)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.best_weight = copy.deepcopy(self.model.state_dict())

    def train(self):
        print(self.device)
        print(next(self.model.parameters()).is_cuda)
        ac_monitor = 0
        
        for k in range(self.epochs):
            epoch_loss = []
            for jj, (sentinel, planet, tsx, lbl) in enumerate(self.train_loader):
                # assert not np.any(np.isnan(x))
                if torch.isnan(sentinel).any():
                    print(f'--sentinel has nan values')
                    sentinel = torch.nan_to_num(sentinel, nan=0.0)
                if torch.isnan(planet).any():
                    print(f'--planet has nan values')
                    planet = torch.nan_to_num(planet, nan=0.0)
                if torch.isnan(tsx).any():
                    # print(f'tsx has nan values')
                    tsx = torch.nan_to_num(tsx, nan=0.0)
                assert not torch.isnan(lbl).any(), f'label has nan values'
                planet = planet.squeeze().float().to(self.device)
                sentinel = sentinel.squeeze().float().to(self.device)
                tsx = tsx.float().squeeze().to(self.device)
                lbl = lbl.float().to(self.device)
                if jj == 0 and k == 0:
                    print(f'Shape of sentinel: {sentinel.shape}')
                    print(f'Shape of planet scope: {planet.shape}')
                    print(f'shape of TSX: {tsx.shape}')
                self.optimizer.zero_grad()
                logits = self.model((planet, sentinel, tsx))
                loss = F.nll_loss(logits, lbl.view(-1).long(), weight=self.class_weight.float())  
                
                epoch_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()
                print(f'epoch {k} ->> step {jj} ->> loss: {loss.item()}', end = '\r', flush=True)
                
            self.tloss.append(np.nanmean(epoch_loss))
            
            v_loss_e = []
            vacc_e = []
            
            with torch.no_grad():
                for kk, (planet, sentinel, tsx, lbl) in enumerate(self.valid_loader):
                    if torch.isnan(sentinel).any():
                        print(f'--sentinel has nan values')
                        sentinel = torch.nan_to_num(sentinel, nan=0.0)
                    if torch.isnan(planet).any():
                        print(f'--planet has nan values')
                        planet = torch.nan_to_num(planet, nan=0.0)
                    if torch.isnan(tsx).any():
                    # print(f'tsx has nan values')
                        tsx = torch.nan_to_num(tsx, nan=0.0)
                    assert not torch.isnan(lbl).any(), f'label has nan values'
                    planet = planet.squeeze().float().to(self.device)
                    
                    sentinel = sentinel.squeeze().float().to(self.device)
                    tsx = tsx.squeeze().float().to(self.device)
                    lbl = lbl.float().to(self.device)
                    vlogs = self.model((planet, sentinel, tsx))
                    vloss = F.nll_loss(vlogs, lbl.view(-1).long(), weight=self.class_weight.float())  
                    vac = self.acc_fn(vlogs.argmax(dim=1), lbl.squeeze().long())
                
                    v_loss_e.append(vloss.item())
                    vacc_e.append(vac)
                    #if kk == 0:
                     #   print(f'Shape of sentinel: {sentinel.shape}')
                     #   print(f'Shape of planet scope: {planet.shape}')
                     #   print(f'shape of TSX: {tsx.shape}')
                    print(f'epoch {k} ->> valstep {kk} ->> vloss: {vloss.item()}', end = '\r', flush=True)
                self.vloss.append(np.nanmean(v_loss_e))
                self.vacc.append(np.nanmean(vacc_e))
            
            #print(f'epoch {k}: tlos {self.tloss[-1]} ->> vloss {np.nanmean(v_loss_e)} ->> acc: {np.nanmean(vacc_e)}')
            
            if k >=1:
                if self.vloss[-1]<self.vloss[-2]:
                    self.best_weight = copy.deepcopy(self.model.state_dict())
                    ac_monitor = 0
                else:
                    ac_monitor+=1

                if k%5 == 0:
                    model_path = f'{self.model_fold}/{self.model_name}_weight.pt'
                    torch.save(self.best_weight, model_path)
                    
    def test(self, read_weight=False, phase='optimize'):
        if read_weight:
            model_path = f'{self.model_fold}/{self.model_name}_weight.pt'
            self.model.load_state_dict(torch.load(model_path))
        else:
            self.model.load_state_dict(self.best_weight)
            self.model.eval()

        control = 0
        PREDS = []
        LBS = []
        
        print('Started Prediction')
        
        for jj, (sentinel, planet, tsx, lbl) in enumerate(self.test_loader):
            if torch.isnan(sentinel).any():
                print(f'--sentinel has nan values')
                sentinel = torch.nan_to_num(sentinel, nan=0.0)
            if torch.isnan(planet).any():
                print(f'--planet has nan values')
                planet = torch.nan_to_num(planet, nan=0.0)
            if torch.isnan(tsx).any():
                    # print(f'tsx has nan values')
                tsx = torch.nan_to_num(tsx, nan=0.0)
            assert not torch.isnan(lbl).any(), f'label has nan values'
            planet = planet.squeeze().float().to(self.device)
            snetinel = sentinel.squeeze().float().to(self.device)
            tsx = tsx.squeeze().float().to(self.device)
            lbl = lbl.float().to(self.device)
            logs = self.model((planet.float(), sentinel.float(), tsx.float())).argmax(-1)
            ac = accuracy_score(np.array(lbl.cpu()).ravel(), np.array(logs.cpu()).ravel())
            f1 = f1_score(np.array(lbl.cpu()).ravel(), np.array(logs.cpu()).ravel(), average='weighted')
            self.accuracy+=ac
            self.f1+=f1
            control+=1
            PREDS.append(np.array(logs.cpu()).reshape(-1,1))
            LBS.append(np.array(lbl.cpu()).reshape(-1,1))
        self.accuracy = self.accuracy/control
        self.f1 = self.f1/control
        
        if phase != 'optimize':
            print('=======================================')
            print(f'Overall test acuracy: {self.accuracy}')
            print(f'Overall f-1 score: {self.f1}')
            print('=======================================')
            text = open(f'{self.rep_fold}/{self.model_name}_test_report.txt','w')
            text.write(f"Overall accuracy: {self.accuracy}\n")
            text.write(f"Micro F-1 score: {self.f1}\n")

            PREDS = np.vstack(tuple(PREDS))
            LBS = np.vstack(tuple(LBS))
            cm_matrics = confusion_matrix(LBS.ravel(), PREDS.ravel(), normalize=None)
            plot_confusion_matrix(cm=cm_matrics,
                                  title='InceptionTime',
                                  cmap=None,
                                  normalize=True,
                                  path=self.rep_fold,
                                  name=f'{self.model_name}')
            
    def predictFullsene(self, save_raster=True, save_array=True):
        self.model.load_state_dict(self.best_weight)
        self.model.eval()
        
        files = f'{self.scene_fold}/Planet/*.npy'
        
        preds = []
        print('Scene prediction started')
        print(f'{len(files)} files obtained for prediction')
        
        for i in range(len(files)):
            a = normalize(torch.from_numpy(np.load(f'{self.scene_fold}/Planet/{i}.npy').astype(float))) 
            b = normalize(torch.from_numpy(np.load(f'{self.scene_fold}/Sentinel/{i}.npy').astype(float)))
            c = normalize(torch.from_numpy(np.load(f'{self.scene_fold}/TSX/{i}.npy').astype(float)))
            inputs = (a.float().to(self.device), b.float().to(self.device), c.float().to(self.device)) # to device
        
            pred = self.model(inputs).argmax(-1)
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
    parser.add_argument('--raster_tempelate', help='Raster path to copy the temelate', type=str)
    parser.add_argument('--batch_size', help='batch size to load task datasets', default=100, type=int, required=False)
    parser.add_argument('--lr', help='learning rate for inner model', default=0.0001, type=float, required=False)
    parser.add_argument('--hdims', help='Hidden inner dimensions of the model', default=10, type=int, required=False)
    parser.add_argument('--nlayer', help='Model number of layers', default=3, type=int, required=False)
    parser.add_argument('--pool', help='Percent layer dropout', default=True, type=bool, required=False)
    parser.add_argument('--dropout', help='Percent layer dropout', default=0.5, type=float, required=False)
    parser.add_argument('--bias', help='Dataset to train', default=True, type=bool, required=False)
    parser.add_argument('--epochs', help='number of epochs to train', default=100, type=int, required=False)
    parser.add_argument('--runner', help='Whether train or test', default='test', type=str, required=False)
    parser.add_argument('--model', help='Model name for optimization and test', default='Transformer', type=str, required=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argumentParser()
    if args.runner == 'train_test':
        loader = customLoader(root=args.data_root,
                              index_root=args.index_fold,
                              batch_size=args.batch_size,
                              phase='train')
        trainer = Trainer(epochs=args.epochs,
                          lr=args.lr,
                          train_loader=loader.trainLoader(),
                          valid_loader=loader.validLoader(),
                          test_loader=loader.testLoader(),
                          class_weight=loader.weight(),
                          num_layer=args.nlayer,
                          h_dim=args.ndims,
                          use_bias=args.bias,
                          with_pool=args.pool,
                          model_name=args.model,
                          model_fold=args.weight_path,
                          rep_fold=args.logpath,
                          pred_scene_fold=args.pred_scene_fold)
        trainer.train()
        trainer.test(phase='test')  
        trainer.predictFullsene()  
        
    elif args.runner == 'optimize':
        loader= customLoader(root=args.data_root,
                index_root=args.index_fold,
                batch_size=args.batch_size,
                phase='train')
        
        lrs = [0.01, 0.001,0.0001,0.00001]
        hdims = [2 **2, 2**3, 2**5, 2**6, 2**7, 2**8]
        nlayers = [2,4,8,16,32]
        biases = [True, False]
        pools = [True, False]
        
        params = {"lrs":[], "hdims":[], "nlayers":[], "biases":[], "pool":[],"accuracy":[]}
        if not os.path.exists(f'{args.log_path}/{args.model}'):
            os.makedirs(f'{args.log_path}/{args.model}')
        text = open(f'{args.log_path}/{args.model}/{args.model}_optimize.txt','w')
        c = 0
        model_weight = None
        tot = len(lrs)*len(hdims)*len(nlayers)*len(biases)*len(pools)
        for lr in lrs:
            for hdim in hdims:
                for nlayer in nlayers:
                    for bias in biases:
                        for pool in pools:
                            print(f'Lr:{lr} \n Hdim: {hdim}\n nlayer: {nlayer}\n, bias: {bias}\n pool:{pool} \n % done: {((c+1)/tot)*100}\n')
                            OPTIMIZER = Trainer(epochs=args.epochs,
                                    lr=args.lr,
                                    train_loader=loader.trainLoader(),
                                    valid_loader=loader.validLoader(),
                                    test_loader=loader.testLoader(),
                                    class_weight=loader.weight,
                                    num_layer=nlayer,
                                    h_dim=hdim,
                                    use_bias=bias,
                                    with_pool=pool,
                                    model_name=args.model,
                                    model_fold=args.weight_path,
                                    rep_fold=args.log_path,
                                    pred_scene_fold=args.pred_scene_fold)
                            OPTIMIZER.train()
                            OPTIMIZER.test(phase=args.runner)
                            text.write(f'Accuracy, {OPTIMIZER.accuracy}, lr: {lr}, hdim: {hdim}, nlayer: {nlayer}, bias {bias}, pool: {pool}\n')
                            params["lrs"].append(lr)
                            params["hdims"].append(hdim)
                            params["nlayers"].append(nlayer)
                            params["biases"].append(bias)
                            params["pool"].append(pool)
                            params["accuracy"].append(OPTIMIZER.accuracy)
                            
                            if (c == 0) or (OPTIMIZER.accuracy >= max(params["accuracy"])):
                                model_weight = OPTIMIZER.best_weight
                            else:
                                pass
                            c+=1
        text.close()
        out_file = f'{args.log_path}/{args.model}/{args.model}_optimize.csv'  # needs optimization
        df = pd.DataFrame.from_dict(params)
        df.to_csv(out_file)
    else:
        raise ValueError(f'provided runner type {args.runner} not known!')

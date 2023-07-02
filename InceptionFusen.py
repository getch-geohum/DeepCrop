import argparse
import itertools
import torch
from torch import nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np 
from inception import Inception, InceptionBlock   # this is mainly for deep learning
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import copy

class Flatten(nn.Module):
    def __init__(self, out_features):
        super(Flatten, self).__init__()
        self.output_dim = out_features

    def forward(self, x):
        return x.view(-1, self.output_dim)
    
class Reshape(nn.Module):
    def __init__(self, out_shape):
        super(Reshape, self).__init__()
        self.out_shape = out_shape

    def forward(self, x):
        return x.view(-1, *self.out_shape)

class FeatureAdaptiveFuse(nn.Module):
    '''Fusion block for the last layer'''
    def __init__(self, dim=-1):
        super(FeatureAdaptiveFuse, self).__init__()
        self.dim = dim
    def forward(self, x, y, m):
        return torch.cat((x, y, m), dim=self.dim)


class CategoricalCrossEntropy(nn.Module):
    def __init__(self, reduce='mean'):
        super(CategoricalCrossEntropy, self).__init__()
        self.reduce=reduce
        
    def forward(self, pred, ref):
        if self.reduce=='mean':
            return (-(pred+1e-5).log() * ref).sum(dim=1).mean()
        else:
            raise ValueError(f'Reduction type {self.reduce} not known')

class Accuracy(nn.Module):
    def __init__(self, reduce='mean'):
        super(Accuracy, self).__init__()
        self.reduce=reduce
        
    def forward(self, pred, ref):
        if self.reduce=='mean':
            return (pred==ref).long().sum()/pred.shape[0]
        else:
            raise ValueError(f'Reduction type {self.reduce} not known')
            
            
class TimeAdaptiveFuse(nn.Module):
    def __init__(self):
        super(TimeAdaptiveFuse, self).__init__()
        
    def forward(self, tensors = [], index=[]):
        shapes = [tensor.shape for tensor in tensors]
        dims = [shape[-1] for shape in shapes]
        ind = dims.index(max(dims)) # where the array with maximum dimension
        fuser = torch.zeros(tensors[ind].shape, dtype=tensors[ind].dtype)  # chek epty data creation system
        for i in range(len(tensors)):
            if i == ind or fuser.shape == tensors[i].shape:
               # print(fuser.shape, tensors[i].shape)
                fuser+=tensors[i]
            else:
                fuser[:,:,index[i]:index[i]+tensors[i].shape[-1]] = fuser[:,:,index[i]:index[i]+tensors[i].shape[-1]] + tensors[i]   # indexing 
        return fuser
    
    
class InceptionFuseNet(nn.Module):
    def __init__(self,
                 in_channels=[4,3,2],
                 n_filters=32,
                 kernels=[5, 7, 9],
                 bottleneck_size=32,
                 with_residual=True,
                 final_features = 11,
                 activation=nn.ReLU(),
                 fuse_stage='middle',
                 inds=[0,6,8]):
        
        super(InceptionFuseNet, self).__init__()
        
        self.in_channels = in_channels
        self.n_filters = n_filters
        self.kernels = kernels
        self.bottleneck_size = bottleneck_size
        self.final_features = final_features
        self.with_residual = with_residual
        self.final_pool =  nn.AdaptiveAvgPool1d(output_size=1)
        self.linear = nn.Linear(in_features=11*32*1, out_features=11)
        self.make_flat = Flatten(out_features=32*11*1)
        self.fuse_stage = fuse_stage
        self.featureFuse = FeatureAdaptiveFuse()
        self.timeFuse = TimeAdaptiveFuse()
        self.activation = activation
        self.final_activation = nn.Softmax(dim=1)
        self.inds = inds
        
        self.inceptionA = [InceptionBlock(
            in_channels=self.in_channels[i],
            n_filters=self.n_filters,
            kernel_sizes=self.kernels,
            bottleneck_channels=self.bottleneck_size,
            use_residual=self.with_residual,
            activation=self.activation) for i in range(len(self.in_channels))]
        
        
        self.inceptionB = InceptionBlock(
            in_channels=self.bottleneck_size*4,
            n_filters=self.n_filters,
            kernel_sizes=self.kernels,
            bottleneck_channels=self.bottleneck_size,
            use_residual=self.with_residual,
            activation=self.activation)
        
        self.finalBlock = nn.Sequential(Flatten(out_features=self.bottleneck_size*4), 
                                        nn.Linear(in_features=self.bottleneck_size*4,
                                                  out_features=self.final_features))
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)

    def forward(self, x, y, z):
        if self.fuse_stage == 'middle':
            x = self.inceptionA[0](x)
            y = self.inceptionA[1](y) 
            z = self.inceptionA[2](z)
            #print('Yessss')
            #print(f'X shape: {x.shape}')
            #print(f'y shape: {y.shape}')
            #print(f'Z shape: {z.shape}')
            
            fused = self.timeFuse([x,y,z], index=self.inds) # based on the time series length
            #print(f'Fused shape :{fused.shape}')
            
            downstage = self.inceptionB(fused)
            #print(f'donwstae shep: {downstage.shape}')
            packt = self.pool(downstage)
            #print(f'pocket shep: {packt.shape}')
            
            outs = self.finalBlock(packt)
            probs = self.final_activation(outs)
            return probs
            
        elif self.fuse_stage == 'last':
            x = self.inceptionA[0](x)
            x = self.inceptionB(x)
            x = self.pool(x)
            
            y = self.inceptionA[1](y)
            y = self.inceptionB(y)
            y = self.pool(y)
            
            z = self.inceptionA[2](z)
            z = self.inceptionB(z)
            z = self.pool(z)
            
            fused = self.featureFuse(x, y, z)
            outs = self.finalBlock(fused)
            probs = self.final_activation(outs)
            return probs
        else:
            raise ValueError(f'Spacified fusion stage {self.fuse_stage} is not known!')
            

class LSTM(nn.Module):
    def __init__(self, num_classes=11, input_size=[4,3,2], hidden_size=256, num_layers=4, proj_size=30, times=[15,9,7]):
        super(LSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.times = times
        self.proj_size = proj_size
        
        self.lstms = [nn.LSTM(input_size=input_size[i], hidden_size=hidden_size,
                              num_layers=num_layers, batch_first=True) for i in range(len(input_size))]
        
        self.projector = nn.Linear(hidden_size, proj_size)  
        self.project2 = nn.Linear(proj_size*sum(times), proj_size)
        
        self.fc = nn.Linear(proj_size, num_classes)
        self.activate = nn.ReLU()

    def forward(self, x, y, z):
        #print(f'X shape : {x.shape}')
        #print(f'Y shape: {y.shape}')
        #print(f'Z shape: {z.shape}')
        
        assert x.shape[0] == y.shape[0] == z.shape[0], 'The shape of the input layers is not the same'
        #h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        #c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        
        x_ula, (h_out, _) = self.lstms[0](x.squeeze())# (h_0, c_0))
        y_ula, (h_out, _) = self.lstms[1](y.squeeze()) #(h_0, c_0))
        z_ula, (h_out, _) = self.lstms[2](z.squeeze()) ## (h_0, c_0))
        
        x_feat = self.projector(x_ula[-1]) # [-1] is mainly for unknown reasons/taking the last feature 
        y_feat = self.projector(y_ula[-1])
        z_feat = self.projector(z_ula[-1])
        
        x_feat = self.activate(x_feat)
        y_feat = self.activate(y_feat)
        z_feat = self.activate(z_feat)
        
        fuser = torch.cat((x_feat, y_feat, z_feat), dim=1).view(-1,sum(self.times)*self.proj_size)
        fused = self.project2(fuser)
        fused = self.activate(fused)

        out = self.fc(fused)
        
        return out


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

        return (sen_sample_t, plan_sample_t,tsx_sample_t, label_sample_t), (sen_sample_v, plan_sample_v,tsx_sample_v, label_sample_v)
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
        
        #print(f'sentinel length: {len(self.snt)}')
        #print(f'planet length: {len(self.plt)}')
        #print(f'TarraSAR-X length: {len(self.tsx)}')
        #print(f'label length: {len(self.lbl)}')
        
        assert len(self.plt) == len(self.snt) == len(self.tsx) == len(self.lbl), 'Length of dataset imagery and labells are not equal.'
        
    def __len__(self):
        return len(self.plt)
    
    def __getitem__(self, idx):
        planet = self.plt[idx]
        sentinel = self.snt[idx]
        terrasar = self.tsx[idx]
        label = self.lbl[idx]
        # print(terrasar.shape)
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
        
        if self.phase == 'train':
            self.train_files, self.valid_files = sampleSplitArray(root=self.root,phase='train')
                                                                 # t_index=None, # self.train_index,
                                                                 # v_index=None, # self.valid_index,
                                                                 # phase='train')
        
        if self.phase == 'test':
            self.test_files = sampleSplitArray(root=self.root,phase='test')
                                               #t_index=None,
                                               #v_index=None,
                                               #phase='test')

    def trainLoader(self):
        train_dataset = SatDataset(self.train_files[0], self.train_files[1], self.train_files[2], self.train_files[3]) 
        return DataLoader(dataset=train_dataset, batch_size = self.batch_size, drop_last=True, num_workers=0, shuffle=True)
    
    def validLoader(self):
        validation_dataset = SatDataset(self.valid_files[0], self.valid_files[1], self.valid_files[2], self.valid_files[3])
        return DataLoader(dataset=validation_dataset, batch_size = self.batch_size, drop_last=True, num_workers=0, shuffle=True)
    
    def testLoader(self):
        test_dataset = SatDataset(self.test_files[0], self.test_files[1], self.test_files[2], self.test_files[3])
        return DataLoader(dataset=test_dataset, batch_size = self.batch_size, drop_last=True, num_workers=0, shuffle=True)
    

class Trainer:
    def __init__(self, model=LSTM().cuda(),epochs=100, lr=0.0001, train_loader=None, valid_loader=None, model_fold=None, rep_fold=None):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.loss_fn = nn.CrossEntropyLoss() #CategoricalCrossEntropy() # reduction
        self.acc_fn = Accuracy()
        self.epochs = epochs
        self.lr = lr
        self.rep_fold = rep_fold
        self.model_fold = model_fold
        
        self.tloss = []
        self.vloss = []
        self.vacc = []
        
        #self.best_weight = copy.deepcopy(self.model.state_dict())
        self.petience = 10  # early stoping after 10 epochs
        self.device = torch.device('cpu')
        self.model = model.train().to('cpu')
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.best_weight = copy.deepcopy(self.model.state_dict())

    def train(self):
        print(self.device)
        print(next(self.model.parameters()).is_cuda)
        ac_monitor = 0
        for k in range(self.epochs):
            epoch_loss = []
            for jj, (planet, sentinel, tsx, lbl) in enumerate(self.train_loader):
                planet = planet.float().to(self.device)
                sentinel = sentinel.float().to(self.device)
                tsx = tsx.float().to(self.device)
                lbl = lbl.float().to(self.device)

                self.optimizer.zero_grad()
                #print(f'label tensor: {lbl.shape}')
                logits = self.model(planet, sentinel, tsx)
                loss = self.loss_fn(logits, lbl.squeeze().long())
                epoch_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()
                print(f'epoch {k} ->> step {jj} ->> loss: {loss.item()}', end = '\r', flush=True)
                
            self.tloss.append(np.nanmean(epoch_loss))
            
            v_loss_e = []
            vacc_e = []

            with torch.no_grad():
                for kk, (planet, sentinel, tsx, lbl) in enumerate(self.valid_loader):
                    planet = planet.float().to(self.device)
                    sentinel = sentinel.float().to(self.device)
                    tsx = tsx.float().to(self.device)
                    lbl = lbl.float().to(self.device)
                    # print(lbl.tolist())
                    vlogs = self.model(planet, sentinel, tsx)
                    vloss = self.loss_fn(vlogs, lbl.squeeze().long())
                    vac = self.acc_fn(vlogs.argmax(dim=1), lbl.squeeze().long())
                
                    v_loss_e.append(vloss.item())
                    vacc_e.append(vac)
                    print(f'epoch {k} ->> valstep {kk} ->> vloss: {vloss.item()}', end = '\r', flush=True)
                self.vloss.append(np.nanmean(v_loss_e))
                self.vacc.append(np.nanmean(vacc_e))
            
            print(f'epoch {k}: tlos {self.tloss[-1]} ->> vloss {np.nanmean(v_loss_e)} ->> acc: {np.nanmean(vacc_e)}')
            
            if k >=1:
                if self.vloss[-1]<self.vloss[-2]:
                    self.best_weight = copy.deepcopy(self.model.state_dict())
                    ac_monitor = 0
                else:
                    ac_monitor+=1

                if k%5 == 0:
                    model_path = f'{self.model_fold}/weight.pt'
                    torch.save(self.best_weight, model_path)
                    
            #if self.petience<ac_monitor:
             ##   acc = f'{self.rep_fold}/vcc.pt'
               # vls = f'{self.rep_fold}/vls.pt'
                #tls = f'{self.rep_fold}/tloss.pt'
                #torch.save(torch.from_numpy(np.array(self.tloss)), tls)
                #torch.save(torch.from_numpy(np.array(self.vloss)), vls)
                #torch.save(torch.from_numpy(np.array(self.vacc)), acc)
                #if not os.path.exists(self.model_fold):
                 #   os.makedirs(self.model_fold, exists_ok=True)
                
                #model_path = f'{self.model_fold}/weight.pt'
                #torch.save(self.best_weight, model_path)
                
               # return None  # just for early stop and return non, can also made to return the model

                
def plot_confusion_matrix(cm,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          path=None):
    target_names = ['Bareland',
                    'Fruits',
                    'Grass',
                    'Guizota',
                    'Lupine',
                    'Maize',
                    'Millet',
                    'Others',
                    'Pepper',
                    'Teff',
                    'Vegetables'
                   ]
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
    print(f'Accuracy is: {accuracy}')
    import pandas as pd
    df = pd.DataFrame(my_array, columns = target_names)
    print(df)

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    fname = f'{path}/figure.png'
    plt.savefig(fname=fname, dpi=350, facecolor='auto', edgecolor='auto', bbox_inches='tight', pad_inches=0.1)
#     plt.show()
    
class Tester:
    def __init__(self, model=LSTM(), test_loader=None, model_fold=None, rep_fold=None):
        self.test_loader = test_loader
        self.model_fold = model_fold
        self.rep_fold = rep_fold
        self.acc_fn = Accuracy()
        self.device = torch.device('cpu') # if torch.cuda.is_available() else 'cpu')
        self.model = model # 
        self.model.load_state_dict(torch.load(f'{self.model_fold}/weight.pt'))# .to('cpu')
        self.AC = []
        self.pred = []
        self.ref = []
        self.softer = torch.nn.LogSoftmax(dim=1)
        
    def test(self):
        self.model.eval()
        for jj, (snetinel, planet, tsx, lbl) in enumerate(self.test_loader):
                planet = planet.float().to(self.device)
                snetinel = snetinel.float().to(self.device)
                tsx = tsx.float().to(self.device)
                lbl = lbl.to(self.device)
                soft_prob = self.model(snetinel, planet, tsx)
                soft_prob = self.softer(soft_prob)
                soft_prob = soft_prob.argmax(dim=1)
                acc = self.acc_fn(soft_prob, lbl.argmax(dim=1))
                self.AC.append(acc)
                self.pred+=soft_prob.view(-1).tolist()
                self.ref+=lbl.argmax(dim=1).view(-1).tolist()
                print(f'{jj} done!', end='\r', flush=True)
        assert len(self.pred) == len(self.ref), 'predicted and referenced files are not the same'
        cmaps = confusion_matrix(np.array(self.pred), np.array(self.ref)) 
        plot_confusion_matrix(cm=cmaps,
                              path=self.rep_fold)

def argumentParser():
    parser = argparse.ArgumentParser(description='model agnostic meta learning implementation both for classic and MAML')
    parser.add_argument('--data_root', help='Root folder that contained all tensors folders', type=str)
    parser.add_argument('--index_root', help='Folder tensor of indexes', type=str)
    parser.add_argument('--weight_path', help='path/folder to save weight', type = str)
    parser.add_argument('--log_path', help='Folder to save the reports and logs', type=str)
    parser.add_argument('--batch_size', help='batch size to load task datasets', default=100, type=int, required=False)
    parser.add_argument('--lr', help='adaptation learning rate for inner model', default=0.0001, type=float, required=False)
    parser.add_argument('--epochs', help='number of epochs to train', default=100, type=int, required=False)
    parser.add_argument('--freq', help='number of repeated experiments', default=4, type=int, required=False)
    parser.add_argument('--runner', help='number of repeated experiments', default='test', type=str, required=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = argumentParser()
    if args.runner == 'train':
        loader = customLoader(root=args.data_root,
                              index_root=args.index_root,
                              batch_size=args.batch_size,
                              phase='train')

        trainer = Trainer(train_loader=loader.trainLoader(),
                          epochs=args.epochs,
                          lr=args.lr,
                          valid_loader=loader.validLoader(),
                          model_fold=args.weight_path,
                          rep_fold=args.log_path)
        trainer.train()

    elif args.runner == 'test':
        loader = customLoader(root=args.data_root,
                              index_root=args.index_root,
                              batch_size=args.batch_size,
                              phase='test')
        tester = Tester(test_loader=loader.testLoader(),
                model_fold=args.weight_path,
                rep_fold=args.log_path)
        tester.test()

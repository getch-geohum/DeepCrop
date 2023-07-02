import argparse
import itertools
import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np 
from inception import Inception, InceptionBlock   # this is mainly for deep learning
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torchmetrics
import copy 

def sampleSplitArray(root, phase='train', data='Sentinel', one_hot=False):
    sen_files = np.array(torch.load(f'{root}/{data}/{phase}/images/points.pt'))
    sen_files = sen_files.reshape(sen_files.shape[0], sen_files.shape[1],1)
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

        return (X_train, y_train), (X_test,y_test),weight
    else:
        return sen_files, label_files, weight
    
def normalize(tensor):
    return (tensor-tensor.min())/(tensor.max()-tensor.min())

def channel_swap(tensor):
    return tensor.permute(1, 0)


class SatDataset(Dataset):
    def __init__(self, plt, lbl, read=False, normalize=True):
        self.plt = plt
        self.lbl = lbl
        self.read = read
        self.normalize = normalize
        
        #print(f'sentinel length: {len(self.plt)}')
        #print(f'label length: {len(self.lbl)}')
        
        assert len(self.plt) == len(self.lbl), 'Length of dataset imagery and labells are not equal.'
        
    def __len__(self):
        return len(self.plt)
    
    def __getitem__(self, idx):
        planet = self.plt[idx]
        label = self.lbl[idx]
        
        if self.read:
            snx = torch.load(planet).permute(1,0)
            lby = torch.load(label)
            if self.normalize:
                return normalize(snx), lby
            else:
                return snx,lby
        else:
            if self.normalize:
                return normalize(planet), label
            else:
                return torch.from_numpy(planet), torch.from_numpy(label)
            
class customLoader:
    def __init__(self, root, batch_size=100, phase='train', data='Sentinel'):
        self.root = root
        self.phase = phase
        self.batch_size = batch_size
        
        if self.phase == 'train':
            self.train_files, self.valid_files, self.weight = sampleSplitArray(root=self.root,
                                                                  phase='train',
                                                                  data=data)
        if self.phase == 'test':
            self.test_files = sampleSplitArray(root=self.root,
                                               phase='test',
                                               data=data)

    def trainLoader(self):
        train_dataset = SatDataset(self.train_files[0], self.train_files[1]) 
        return DataLoader(dataset=train_dataset, batch_size = self.batch_size, drop_last=True, num_workers=0, shuffle=True)
    
    def validLoader(self):
        validation_dataset = SatDataset(self.valid_files[0], self.valid_files[1])
        return DataLoader(dataset=validation_dataset, batch_size = self.batch_size, drop_last=True, num_workers=0, shuffle=True)
    
    def testLoader(self):
        test_dataset = SatDataset(self.test_files[0], self.test_files[1])
        return DataLoader(dataset=test_dataset, batch_size = self.batch_size, drop_last=True, num_workers=0, shuffle=True)
    

class Accuracy(nn.Module):
    def __init__(self, reduce='mean'):
        super(Accuracy, self).__init__()
        self.reduce=reduce
        
    def forward(self, pred, ref):
        if self.reduce=='mean':
            return (pred==ref).long().sum()/pred.shape[0]
        else:
            raise ValueError(f'Reduction type {self.reduce} not known')


class VanilaLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, neck_size):
        super(VanilaLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, bias=False, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size*2,neck_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.8)
        self.fc2 = nn.Linear(neck_size, num_classes)
        self.logmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        out, _ = self.lstm(x)
        out = self.relu(out[:,-1,:])
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.logmax(out)
        return out
    
class Trainer:
    def __init__(self,epochs=100, lr=0.0001, train_loader=None, valid_loader=None, model_fold=None, rep_fold=None, data='Sentinel', num_class=11, weight=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        #self.loss_fn =  # nn.CrossEntropyLoss(reduction="mean") # CategoricalCrossEntropy() # reduction
        self.acc_fn = torchmetrics.Accuracy().to(self.device)
        self.epochs = epochs
        self.lr = lr
        self.weight = weight.cuda()
        self.rep_fold = rep_fold
        self.model_fold = model_fold

        self.tloss = []
        self.vloss = []
        self.vacc = []
        self.tolerance = 10
        self.ac_monitor = 0
        
        self.petience = 10  # early stoping after 10 epochs
        if data == "Sentinel":
            input_size = 1
        elif data == 'TSX':
            input_size = 1
        elif data == 'Planet':
            input_size = 1
        else:
            raise ValeError(f'The specified dataset {data} not known')
        self.model = VanilaLSTM(input_size=input_size, hidden_size=12, num_layers=4, num_classes=num_class,neck_size=10)
        self.model.train()
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.best_weight = copy.deepcopy(self.model.state_dict())

    def train(self):
        ac_monitor = 0
        for k in range(self.epochs):
            epoch_loss = []
            for jj, (sentinel, lbl) in enumerate(self.train_loader):
                sentinel = sentinel.to(self.device)
                lbl = lbl.to(self.device)
            
                self.optimizer.zero_grad()
                logits = self.model(sentinel.float())
                loss = F.nll_loss(logits, lbl.view(-1).long(), weight=self.weight.float())
                # loss = self.loss_fn(logits, lbl.view(-1).long())
                epoch_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()
                print(f'epoch {k} ->> step {jj} ->> loss: {loss.item()}', end = '\r', flush=True)
    
            self.tloss.append(np.nanmean(epoch_loss))
            
            v_loss_e = []
            vacc_e = []
            
            for kk, (sentinel, lbl) in enumerate(self.valid_loader):
                sentinel = sentinel.to(self.device)
                lbl = lbl.to(self.device)

                vlogs = self.model(sentinel.float())
                vloss = F.nll_loss(vlogs, lbl.view(-1).long(), weight=self.weight.float())
                #vloss = self.loss_fn(vlogs, lbl.view(-1).long())

                vac = self.acc_fn(vlogs.argmax(dim=-1), lbl.view(-1).long())

                v_loss_e.append(vloss.item())
                vacc_e.append(vac.item())
            self.vloss.append(np.nanmean(v_loss_e))
            self.vacc.append(np.nanmean(vacc_e))
            
            print(f'epoch {k} ->> vloss {np.nanmean(v_loss_e)} ->> acc: {np.nanmean(vacc_e)}')
            
            if k >=1:
                if self.vloss[-1]<self.vloss[-2]:
                    self.best_weight = copy.deepcopy(self.model.state_dict())
                    self.ac_monitor = 0
                else:
                    self.ac_monitor+=1

                if k%5 == 0:
                    model_path = f'{self.model_fold}/weight.pt'
                    torch.save(self.best_weight, model_path)
                    
            if self.tolerance<self.ac_monitor:
                acc = f'{self.rep_fold}/vcc.pt'
                vls = f'{self.rep_fold}/vls.pt'
                tls = f'{self.rep_fold}/tloss.pt'
                torch.save(torch.from_numpy(np.array(self.tloss)), tls)
                torch.save(torch.from_numpy(np.array(self.vloss)), vls)
                torch.save(torch.from_numpy(np.array(self.vacc)), acc)
                
                model_path = f'{self.model_fold}/weight.pt'
                torch.save(self.best_weight, model_path)
                
                return None  # just for early stop and return non, can also made to return the model

                
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

    
class Tester:
    def __init__(self, test_loader=None, model_fold=None, rep_fold=None, data='Sentinel'):
        self.test_loader = test_loader
        self.model_fold = model_fold
        self.rep_fold = rep_fold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.acc_fn = torchmetrics.Accuracy(num_classes=11).to(self.device) # Accuracy()
        if data == "Sentinel":
            input_size = 1
        elif data == 'TSX':
            input_size = 1
        elif data == 'Planet':
            input_size = 1
        else:
            raise ValeError(f'The specified dataset {data} not known')
        self.model = VanilaLSTM(input_size=input_size, hidden_size=256, num_layers=64, num_classes=11,neck_size=100)
        self.model.load_state_dict(torch.load(f'{self.model_fold}/weight.pt'))
        self.model.to(self.device)
        self.AC = []
        self.pred = []
        self.ref = []
        
    def test(self):
        for jj, (sentinel, lbl) in enumerate(self.test_loader):
                sentinel = sentinel.to(self.device)
                lbl = lbl.to(self.device)
                soft_prob = self.model(sentinel.float())
                soft_prob = soft_prob.argmax(dim=1)
                acc = self.acc_fn(soft_prob, lbl.view(-1).long())
                print(jj, ': ', acc.item())
                self.AC.append(acc)
                self.pred+=soft_prob.view(-1).tolist()
                self.ref+=lbl.view(-1).tolist()
        assert len(self.pred) == len(self.ref), 'predicted and referenced files are not the same'

        cmaps = confusion_matrix(np.array(self.pred), np.array(self.ref))
        #plot_confusion_matrix(cm=cmaps,
        #                      path=self.rep_fold)
        print(cmaps)

def argumentParser():
    parser = argparse.ArgumentParser(description='model agnostic meta learning implementation both for classic and MAML')
    parser.add_argument('--data_root', help='Root folder that contained all tensors folders', type=str)
    #parser.add_argument('--index_root', help='Folder tensor of indexes', type=str)
    parser.add_argument('--weight_path', help='path/folder to save weight', type = str)
    parser.add_argument('--log_path', help='Folder to save the reports and logs', type=str)
    parser.add_argument('--batch_size', help='batch size to load task datasets', default=100, type=int, required=False)
    parser.add_argument('--lr', help='adaptation learning rate for inner model', default=0.0001, type=float, required=False)
    parser.add_argument('--epochs', help='number of epochs to train', default=100, type=int, required=False)
    parser.add_argument('--freq', help='number of repeated experiments', default=4, type=int, required=False)
    parser.add_argument('--runner', help='number of repeated experiments', default='test', type=str, required=False)
    parser.add_argument('--data', help='Dataset to train', default='Sentinel', type=str, required=False)
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = argumentParser()
    if args.runner == 'train':
        loader = customLoader(root=args.data_root,
                batch_size=args.batch_size,
                phase='train',
                data = args.data)

        trainer = Trainer(train_loader=loader.trainLoader(),
                epochs=args.epochs,
                lr=args.lr,
                valid_loader=loader.validLoader(),
                model_fold=args.weight_path,
                rep_fold=args.log_path,
                data=args.data, weight=loader.weight)
        trainer.train()

    elif args.runner == 'test':
        loader = customLoader(root=args.data_root,
                batch_size=args.batch_size,
                phase='test',
                data=args.data)
        tester = Tester(test_loader=loader.testLoader(),
                model_fold=args.weight_path,
                rep_fold=args.log_path,
                data=args.data)
        tester.test()

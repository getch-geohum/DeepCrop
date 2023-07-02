import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch

class InceptionTime(nn.Module):
    def __init__(self,num_classes, input_dim=1,num_layers=6, hidden_dims=128,use_bias=False, use_residual= True, device=torch.device("cpu"), pool=True):
        super(InceptionTime, self).__init__()
        
        self.pool = pool
        self.num_layers = num_layers
        self.use_residual = use_residual
        self.inception_modules_list = [InceptionModule(input_dim = input_dim, kernel_size=40, num_filters=hidden_dims//4,
                                                       use_bias=use_bias, device=device)]
        for i in range(num_layers-1):
            self.inception_modules_list.append(InceptionModule(input_dim = hidden_dims, kernel_size=40, num_filters=hidden_dims//4,
                                                       use_bias=use_bias, device=device))
        self.shortcut_layer_list = [ShortcutLayer(input_dim,hidden_dims,stride = 1, bias = False)]
        for i in range(num_layers//3):
            self.shortcut_layer_list.append(ShortcutLayer(hidden_dims,hidden_dims,stride = 1, bias = False))
            
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.to(device)    

    def forward(self,x):
        x = x.transpose(1,2)
        input_res = x
        
        for d in range(self.num_layers):
            x = self.inception_modules_list[d](x)
            #print(x.shape)

            if self.use_residual and d % 3 == 2:
                x = self.shortcut_layer_list[d//3](input_res, x)
                input_res = x
        if self.pool:
            x = self.avgpool(x).squeeze(2)
        return x #  logprobabilities
    
        
class InceptionModule(nn.Module):
    def __init__(self,input_dim=32, kernel_size=40, num_filters= 32, residual=False, use_bias=False, device=torch.device("cpu")):
        super(InceptionModule, self).__init__()

        self.residual = residual

        self.bottleneck = nn.Conv1d(input_dim, num_filters , kernel_size = 1, stride=1, padding= 0,bias=use_bias)
        
        kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]
        self.convolutions = [nn.Conv1d(num_filters, num_filters, kernel_size=kernel_size+1, stride=1, bias= False, padding=kernel_size//2).to(device) for kernel_size in kernel_size_s] #padding is 1 instead of kernel_size//2
        
        self.pool_conv = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(input_dim, num_filters,kernel_size=1, stride = 1,padding=0, bias=use_bias) 
        )

        self.bn_relu = nn.Sequential(
            nn.BatchNorm1d(num_filters*4),
            nn.ReLU()
        )

        self.to(device)


    def forward(self, input_tensor):
        # collapse feature dimension

        input_inception = self.bottleneck(input_tensor)
        features = [conv(input_inception) for conv in self.convolutions]
        features.append(self.pool_conv(input_tensor.contiguous()))
        features = torch.cat(features, dim=1) 
        features = self.bn_relu(features)
        
        return features

class ShortcutLayer(nn.Module):
    def __init__(self, in_planes, out_planes, stride, bias):
        super(ShortcutLayer, self).__init__()
        self.sc = nn.Sequential(nn.Conv1d(in_channels=in_planes,
                                          out_channels=out_planes,
                                          kernel_size=1,
                                          stride=stride,
                                          bias=bias),
                                nn.BatchNorm1d(num_features=out_planes))
        self.relu = nn.ReLU()

    def forward(self, input_tensor, out_tensor):
        x = out_tensor + self.sc(input_tensor)
        x = self.relu(x)
        return x
    
    
class InceptionFuseNet(nn.Module):
    def __init__(self, num_classes,
            input_dim=[1, 3,3],
            num_layers=1,
            hidden_dims=64,
            use_bias=False,
            use_residual= True,
            device=torch.device("cpu"),
            with_pool=True):
        super(InceptionFuseNet, self).__init__()
        self.with_pool = with_pool
        self.pool = nn.AdaptiveAvgPool1d(1)
        if with_pool:
            self.outlinear = nn.Linear(hidden_dims*3, num_classes)
        else:
            self.outlinear = nn.Linear(hidden_dims, num_classes)
        self.encoders = [InceptionTime(num_classes=num_classes,
                                       input_dim=input_dim[i],
                                       num_layers=num_layers,
                                       hidden_dims=hidden_dims,
                                       use_bias=use_bias,
                                       use_residual=use_residual,
                                       pool=with_pool) for i in range(len(input_dim))]
        
        
        
    def forward(self, vals):
        encods = []
        for j, val in enumerate(vals):
            xx = self.encoders[j](val)
            encods.append(xx)
        encods = torch.cat(encods, dim=-1)
        #print(f'Encodes shape before pool: {encods.shape}')
        if not self.with_pool:
            encods = self.pool(encods).squeeze(2)
            
        #print(f'Encodes shape after pool: {encods.shape}')
        encods = self.outlinear(encods)
        logs = F.log_softmax(encods, dim=-1)
        #print(f'Log probabilities shape: {logs.shape}')
        return logs 

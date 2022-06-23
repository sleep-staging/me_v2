#%%
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
#%%
from .resnet1d import BaseNet

# Residual Block
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=False, pooling=False):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ELU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.maxpool = nn.MaxPool2d(2, stride=2) 
        self.downsample = nn.Sequential(
           nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
           nn.BatchNorm2d(out_channels)
        )
        self.downsampleOrNot = downsample
        self.pooling = pooling
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsampleOrNot:
            residual = self.downsample(x)
        out += residual
        if self.pooling:
            out = self.maxpool(out)
        out = self.dropout(out)
        return out

class CNNEncoder2D_SLEEP(nn.Module):
    def __init__(self, n_dim):
        super(CNNEncoder2D_SLEEP, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 6, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(6),
            nn.ELU(inplace=True),
        )
        self.conv2 = ResBlock(6, 8, 2, True, False)
        self.conv3 = ResBlock(8, 16, 2, True, True)
        self.conv4 = ResBlock(16, 32, 2, True, True)
        self.n_dim = n_dim

        self.fc = nn.Sequential(
            nn.Linear(128, self.n_dim, bias=True),
            nn.ReLU(),
            nn.Linear(self.n_dim, self.n_dim, bias=True),
        )

        self.sup = nn.Sequential(
            nn.Linear(128, 32, bias=True),
            nn.ReLU(),
            nn.Linear(32, 5, bias=True),
        )

        self.byol_mapping = nn.Sequential(
            nn.Linear(128, self.n_dim, bias=True),
            nn.ReLU(),
            nn.Linear(self.n_dim, self.n_dim, bias=True),
        )

    def torch_stft(self, X_train):
        signal = []

        for s in range(X_train.shape[1]):
            spectral = torch.stft(X_train[:, s, :],
                n_fft = 256,
                hop_length = 256 * 1 // 4,
                center = False,
                onesided = True,
                return_complex=False)
            signal.append(spectral)
        
        signal1 = torch.stack(signal)[:, :, :, :, 0].permute(1, 0, 2, 3) # amplitude (B, 2, H, W)
        signal2 = torch.stack(signal)[:, :, :, :, 1].permute(1, 0, 2, 3) # phase (B, 2, H, W)

        return torch.cat([torch.log(torch.abs(signal1) + 1e-8), torch.log(torch.abs(signal2) + 1e-8)], dim=1) # (B, 4, H, W)
        # 0, 1 for 1st channel, 2, 3 for 2nd channel

    def forward(self, x, simsiam=False, mid=False, byol=False, sup=False):
        # Inputs -> (B, 2, 3000)
        x = self.torch_stft(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x) # (B, 32, 4, 1)

        x = x.reshape(x.shape[0], -1)

        if sup:
            return self.sup(x)
        elif simsiam:
            return x, self.fc(x)
        elif mid:
            return x
        elif byol:
            x = self.fc(x)
            x = self.byol_mapping(x)
            return x
        else:
            x = self.fc(x)
            return x
#%%
class encoder(nn.Module):

    def __init__(self,config):
        super(encoder,self).__init__()
        self.time_model = BaseNet()
        self.attention = attention(config)
        
    def forward(self, x): 

        time = self.time_model(x)

        time_feats = self.attention(time)

        return time_feats


class attention(nn.Module):
    def __init__(self,config):
        super(attention,self).__init__()
        self.att_dim =256
        self.W = nn.Parameter(torch.randn(256, self.att_dim))
        self.V = nn.Parameter(torch.randn(self.att_dim, 1))
        self.scale = self.att_dim**-0.5
    def forward(self,x):
        x = x.permute(0, 2, 1)
        e = torch.matmul(x, self.W)
        e = torch.matmul(torch.tanh(e), self.V)
        e = e*self.scale
        n1 = torch.exp(e)
        n2 = torch.sum(torch.exp(e), 1, keepdim=True)
        alpha = torch.div(n1, n2)
        x = torch.sum(torch.mul(alpha, x), 1)
        return x


class projection_head(nn.Module):

    def __init__(self,config,input_dim=256):
        super(projection_head,self).__init__()
        self.config = config
        self.projection_head = nn.Sequential(
                nn.Linear(input_dim,config.tc_hidden_dim),
                nn.BatchNorm1d(config.tc_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(config.tc_hidden_dim,config.tc_hidden_dim))
 
    def forward(self,x):
        x = x.reshape(x.shape[0],-1)
        x = self.projection_head(x)
        return x

class sleep_model(nn.Module):

    def __init__(self,config):
        super(sleep_model,self).__init__()
        self.eeg_encoder= encoder(config)
        self.weak_pj1 = projection_head(config)
        self.pred = projection_head(config,input_dim=config.tc_hidden_dim)

        self.wandb = config.wandb

    def forward(self,weak_dat,strong_dat):
        weak_eeg_dat = weak_dat.float()

        strong_eeg_dat = strong_dat.float()
        
        weak_time_feats= self.eeg_encoder(weak_eeg_dat)
        strong_time_feats= self.eeg_encoder(strong_eeg_dat)

        proj1 = self.weak_pj1(weak_time_feats)
        pred1 = self.pred(proj1)
        proj2 = self.weak_pj1(strong_time_feats)
        pred2 = self.pred(proj2)
        


        return pred1,proj1,pred2,proj2

class contrast_loss(nn.Module):

    def __init__(self,config):

        super(contrast_loss,self).__init__()
        self.model = sleep_model(config)
        self.T = config.temperature
        self.bn = nn.BatchNorm1d(config.tc_hidden_dim//2,affine=False)
        self.bn2 = nn.BatchNorm1d(config.tc_hidden_dim//2,affine=False)
        self.bs = config.batch_size
        self.lambd = 0.05
        self.mse = nn.MSELoss(reduction='mean')
        self.wandb = config.wandb

    def loss(self, p1, p2, z1, z2):

        # L2 normalize
        p1 = F.normalize(p1, p=2, dim=1)
        p2 = F.normalize(p2, p=2, dim=1)
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        
        # mutual prediction
        l_pos1 = torch.einsum('nc,nc->n', [p1, z2.detach()]).unsqueeze(-1) # Using stop gradients, z2 doesn't involve in the computation, only p1 is responsible for learning
        l_pos2 = torch.einsum('nc,nc->n', [p2, z1.detach()]).unsqueeze(-1)

        loss = - (l_pos1.mean() + l_pos2.mean()) / 2 # using mean
                
        return loss

#%%

    def forward(self,weak,strong,epoch):
        pred1,proj1,pred2,proj2 = self.model(weak,strong)
        l1 = self.loss(pred1,pred2,proj1,proj2)
        tot_loss = l1

        return tot_loss,l1.item(),0,0,0


#%%
class ft_loss(nn.Module):

    def __init__(self,chkpoint_pth,config,device):

        super(ft_loss,self).__init__()
        self.eeg_encoder = encoder(config)
        
        chkpoint = torch.load(chkpoint_pth,map_location=device)
        eeg_dict = chkpoint['eeg_model_state_dict']

        self.eeg_encoder.load_state_dict(eeg_dict)

        for p in self.eeg_encoder.parameters():
            p.requires_grad=False

        self.time_model = self.eeg_encoder.time_model
        self.attention = self.eeg_encoder.attention
        self.lin = nn.Linear(256,5)

    def forward(self,time_dat):

        time_feats= self.time_model(time_dat)
        time_feats = self.attention(time_feats)
        x = self.lin(time_feats)
        return x 

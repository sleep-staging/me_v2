#%%
import torch.nn as nn
import torch.nn.functional as F
import torch
#%%
from .resnet1d import BaseNet

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

        self.q_encoder = encoder(config)
        self.k_encoder = encoder(config)

        for param_q, param_k in zip(self.q_encoder.parameters(), self.k_encoder.parameters()):
            param_k.data.copy_(param_q.data) 
            param_k.requires_grad = False  # not update by gradient

        self.q_proj  = projection_head(config)
        self.k_proj = projection_head(config)

        for param_q, param_k in zip(self.q_proj.parameters(), self.k_proj.parameters()):
            param_k.data.copy_(param_q.data) 
            param_k.requires_grad = False  # not update by gradient

        self.wandb = config.wandb


    def forward(self,weak_dat,strong_dat):

        weak_eeg_dat = weak_dat.float()
        strong_eeg_dat = strong_dat.float()

        anchor = self.q_encoder(weak_eeg_dat)
        anchor = self.q_proj(anchor)
        positive = self.k_encoder(strong_eeg_dat)
        positive = self.k_proj(positive)

        return anchor,positive

class contrast_loss(nn.Module):

    def __init__(self,config):

        super(contrast_loss,self).__init__()
        self.model = sleep_model(config)
        self.T = config.temperature
        self.wandb = config.wandb
        self.config = config
        self.n_queue = 4096 # SIZE of the dictionary queue
        self.queue = torch.rand((self.n_queue, 128), dtype = torch.float).to(config.device)
        self.ptr = 0
        
    def loss(self, anchor, positive, queue):
        
        # L2 normalize
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        queue = F.normalize(queue, p=2, dim=1)

        # positive logits: Nx1, negative logits: NxK
        l_pos = torch.einsum('nc,nc->n', [anchor, positive]).unsqueeze(-1)
        l_neg = torch.einsum('nc,kc->nk', [anchor, queue])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.config.device)

        # loss
        loss = F.cross_entropy(logits, labels)
        
        return loss # mean
#%%

    def forward(self,weak,strong,epoch):
        anchor,positive= self.model(weak,strong)
        l1 = self.loss(anchor,positive,self.queue)
        
        # Updating queue
        if self.queue.shape[0] == self.n_queue:
            self.queue = torch.roll(self.queue, -positive.shape[0], 0)
            self.queue[-positive.shape[0]:] = positive
        else:
            self.queue[self.ptr: self.ptr+positive.shape[0]] = positive
            self.ptr += positive.shape[0]
        
        return l1,l1.item(),0,0,0


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

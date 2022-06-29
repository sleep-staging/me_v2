import torch.nn as nn
import torch.nn.functional as F
import torch

from .resnet1d import BaseNet


class attention(nn.Module):
        
    def __init__(self, n_dim=256):
        super(attention,self).__init__()
        self.att_dim = n_dim
        self.W = nn.Parameter(torch.randn(n_dim, self.att_dim))
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
    
    
class encoder(nn.Module):

    def __init__(self):
        super(encoder,self).__init__()
        self.time_model = BaseNet()
        self.attention = attention()
        
    def forward(self, x): 
        x = self.time_model(x)
        x = self.attention(x)
        return x


class projection_head(nn.Module):

    def __init__(self, config, input_dim=256):
        super(projection_head,self).__init__()
        self.config = config
        
        self.projection_head = nn.Sequential(
                    nn.Linear(input_dim, config.proj_dim, bias=True),
                    nn.BatchNorm1d(config.proj_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(config.proj_dim, config.proj_dim, bias=True)
                    nn.BatchNorm1d(config.proj_dim),
                )
 
    def forward(self,x):
        x = x.reshape(x.shape[0], -1) # B, 128
        x = self.projection_head(x)
        return x

class predictor_head(nn.Module):

    def __init__(self, config):
        super(predictor_head,self).__init__()
        self.config = config
        
        self.projection_head = nn.Sequential(
                    nn.Linear(config.proj_dim, config.proj_dim, bias=True),
                    nn.BatchNorm1d(config.proj_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(config.proj_dim, config.proj_dim, bias=True)
                )
 
    def forward(self,x):
        x = x.reshape(x.shape[0], -1) # B, 128
        x = self.projection_head(x)
        return x

class sleep_model(nn.Module):

    def __init__(self, config):
        super(sleep_model,self).__init__()
        
        self.eeg_encoder= encoder()
        self.proj = projection_head(config)
        self.pred = predictor_head(config)

    def forward(self, weak_data, strong_data):
        weak_data= self.eeg_encoder(weak_data)
        strong_data= self.eeg_encoder(strong_data)

        proj1 = self.proj(weak_data)
        pred1 = self.pred(proj1)
       
        proj2 = self.proj(strong_data)
        pred2 = self.pred(proj2)

        return pred1, proj1, pred2, proj2

class contrast_loss(nn.Module):

    def __init__(self,config):

        super(contrast_loss,self).__init__()
        self.model = sleep_model(config)
        self.T = config.temperature
    

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

    def forward(self, weak, strong):
        pred1, proj1, pred2, proj2 = self.model(weak, strong)
        loss = self.loss(pred1, pred2, proj1, proj2)
        return loss
    
    
class ft_loss(nn.Module):

    def __init__(self, chkpoint_pth, config, device):

        super(ft_loss,self).__init__()
        self.eeg_encoder = encoder()
        
        chkpoint = torch.load(chkpoint_pth, map_location=device)
        eeg_dict = chkpoint['eeg_model_state_dict']

        self.eeg_encoder.load_state_dict(eeg_dict)

        for p in self.eeg_encoder.parameters():
            p.requires_grad=False

        self.time_model = self.eeg_encoder.time_model
        self.attention = self.eeg_encoder.attention
        self.lin = nn.Linear(256, 5)

    def forward(self, x):

        x= self.time_model(x)
        x = self.attention(x)
        x = self.lin(x)
        return x 

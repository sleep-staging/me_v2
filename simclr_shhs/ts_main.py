#%%
import wandb
import numpy as np
import pytorch_lightning as pl
import os
import torch
from sklearn.model_selection import KFold
from pytorch_lightning.callbacks import LearningRateMonitor
from data_preprocessing.dataloader import data_generator
from trainer import sleep_ft,sleep_pretrain
from config import Config

path = "/scratch/new_shhs/"


training_mode = 'ss'
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

#%%
# for self supervised training
if training_mode == 'ss':
    name = 'simclr_shhs_diff_pj_temp_0_5'
    ss_wandb = wandb.init(project='new_baselines',name=name,notes='have used spectrogram and time with 3 contrastive loss',save_code=True,entity='sleep-staging')
    config = Config(ss_wandb)
    ss_wandb.save('/home/vamsi81523/v2_new_models/simclr_shhs//config.py')
    ss_wandb.save('/home/vamsi81523/v2_new_models/simclr_shhs//trainer.py')
    ss_wandb.save('/home/vamsi81523/v2_new_models/simclr_shhs//data_preprocessing/*')
    ss_wandb.save('/home/vamsi81523/v2_new_models/simclr_shhs//models/*')
    print("Loading")
    dataloader = data_generator(os.path.join(path,'pretext'),config)
    print("Done")
    #%%
    model = sleep_pretrain(config,name,dataloader,ss_wandb)
    print('Model Loaded')
    #ss_wandb.watch([model],log='all',log_freq=500)
    model.fit()
    ss_wandb.finish()

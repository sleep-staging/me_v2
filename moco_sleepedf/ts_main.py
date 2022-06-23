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

path = "/scratch/SLEEP_data/"


training_mode = 'ss'
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

#%%
# for self supervised training
if training_mode == 'ss':
    name = 'moco_sleepedf'
    ss_wandb = wandb.init(project='new_baselines',name=name,notes='normalized each recording',save_code=True,entity='sleep-staging')
    config = Config(ss_wandb)
    ss_wandb.save('/home/vamsi81523/v2_new_models/moco_sleepedf//config.py')
    ss_wandb.save('/home/vamsi81523/v2_new_models/moco_sleepedf//trainer.py')
    ss_wandb.save('/home/vamsi81523/v2_new_models/moco_sleepedf//data_preprocessing/*')
    ss_wandb.save('/home/vamsi81523/v2_new_models/moco_sleepedf//models/*')
    print("Loading")
    dataloader = data_generator(os.path.join(path,'pretext'),config)
    print("Done")
    #%%
    model = sleep_pretrain(config,name,dataloader,ss_wandb)
    print('Model Loaded')
    model.fit()
    ss_wandb.finish()

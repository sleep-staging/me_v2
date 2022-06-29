import wandb
import numpy as np
import os
import torch
from data_preprocessing.dataloader import data_generator
from trainer import sleep_pretrain
from config import Config

path = "/scratch/new_shhs/"

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


name = 'simsiam_shhs_final'
ss_wandb = wandb.init(project='me_v2',name=name,notes='normalized each recording',save_code=True,entity='sleep-staging')
config = Config(ss_wandb)
ss_wandb.save('/home/vamsi81523/v2_new_models/simsiam_shhs//config.py')
ss_wandb.save('/home/vamsi81523/v2_new_models/simsiam_shhs//trainer.py')
ss_wandb.save('/home/vamsi81523/v2_new_models/simsiam_shhs//data_preprocessing/*')
ss_wandb.save('/home/vamsi81523/v2_new_models/simsiam_shhs//models/*')
print("Loading")
dataloader = data_generator(os.path.join(path,'pretext'),config)
print("Done")

model = sleep_pretrain(config,name,dataloader,ss_wandb)
print('Model Loaded')
#ss_wandb.watch([model],log='all',log_freq=500)
model.fit()
ss_wandb.finish()

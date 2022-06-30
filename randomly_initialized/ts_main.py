import wandb
import numpy as np
import os
import torch
from data_preprocessing.dataloader import data_generator
from trainer import sleep_pretrain
from config import Config

path = "/scratch/SLEEP_data/"

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


name = 'randomly_initialized (without es)'
ss_wandb = wandb.init(project='me_v2',name=name ,notes='normalized recording wise',save_code=True,entity='sleep-staging')
config = Config(ss_wandb)

ss_wandb.save('/home/vamsi81523/me_v2/randomly_initialized//config.py')
ss_wandb.save('/home/vamsi81523/me_v2/randomly_initialized//trainer.py')
ss_wandb.save('/home/vamsi81523/me_v2/randomly_initialized//data_preprocessing/*')
ss_wandb.save('/home/vamsi81523/me_v2/randomly_initialized//models/*')

print("Loading dataloader")
dataloader = data_generator(os.path.join(path,'pretext'),config)
print("Done")

model = sleep_pretrain(config,name,dataloader,ss_wandb)
print('Model Loaded')
model.fit()
ss_wandb.finish()

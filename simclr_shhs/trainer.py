import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from config import Config
from torchmetrics.functional import accuracy,f1,cohen_kappa
from models.model import contrast_loss,ft_loss
from sklearn.metrics import ConfusionMatrixDisplay,balanced_accuracy_score
from data_preprocessing.dataloader import TuneDataset
from sklearn.model_selection import KFold
from tqdm import tqdm
from torch.utils.data import DataLoader

config = Config()

class sleep_pretrain(nn.Module):

    def __init__(self,config,name,dataloader,wandb_logger):
        super(sleep_pretrain,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = contrast_loss(config)
        self.model = self.model.to(self.device)
        self.config = config
        self.weight_decay = 3e-5
        self.batch_size = config.batch_size
        self.name = name
        self.dataloader = dataloader
        self.loggr = wandb_logger
        self.optimizer = torch.optim.Adam(self.model.parameters(),self.config.lr,betas=(self.config.beta1,self.config.beta2),weight_decay=self.weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer,mode='min',patience=5,factor=0.2) 
        self.epochs = config.num_epoch
        self.ft_epochs = config.num_ft_epoch

        self.max_f1 = 0
        self.max_mean_f1 = 0
        self.max_kappa = 0
        self.max_bal_acc = 0
        self.max_acc = 0

        TEST_FILE = os.listdir(os.path.join(self.config.sleepedf_path, "test"))
        TEST_FILE = [os.path.join(self.config.sleepedf_path, "test", f) for f in TEST_FILE]
        self.test_subjects = [np.load(file) for file in TEST_FILE]

    def training_step(self,batch,batch_idx):
        weak,strong= batch
        weak,strong = weak.to(self.device),strong.to(self.device)
        loss = self.model(weak,strong,self.current_epoch)
        return loss
    
    def training_epoch_end(self,outputs):
        epoch_loss = torch.hstack([torch.tensor(x) for x in outputs['loss']]).mean()
        self.loggr.log({'Epoch Loss':epoch_loss,
            "LR":self.scheduler.optimizer.param_groups[0]['lr'],
            "Epoch":self.current_epoch})
        self.scheduler.step(epoch_loss)
        return epoch_loss

    def on_epoch_end(self):
        chkpoint = {'eeg_model_state_dict':self.model.model.eeg_encoder.state_dict()}
        torch.save(chkpoint, os.path.join(config.exp_path, self.name+'.pt'))
        self.loggr.save(os.path.join(config.exp_path, self.name+'.pt'))
        full_chkpoint = {'model_state_dict':self.model.state_dict(),'epoch':self.current_epoch}
        torch.save(full_chkpoint, os.path.join(config.exp_path, self.name+"_full"+'.pt'))
        return None

    def ft_fun(self,file_name,epoch,train_dl,valid_dl,split):
        sleep_eval = sleep_ft(config.exp_path+"/"+self.name+'.pt',self.config,train_dl,valid_dl,epoch,self.loggr)
        f1,mean_f1,kappa,bal_acc,acc = sleep_eval.fit()

        return f1,mean_f1,kappa,bal_acc,acc

    def do_kfold(self):
        kfold = KFold(n_splits=5, shuffle=True, random_state=1234)

        k_f1,k_mean_f1,k_kappa,k_bal_acc,k_acc = 0,0,0,0,0
        split=1
        for train_idx, test_idx in kfold.split(self.test_subjects):

            test_subjects_train = [self.test_subjects[i] for i in train_idx]
            test_subjects_test = [self.test_subjects[i] for i in test_idx]

            train_loader = DataLoader(TuneDataset(test_subjects_train), batch_size=self.config.batch_size, shuffle=True)
            test_loader = DataLoader(TuneDataset(test_subjects_test), batch_size=self.config.batch_size, shuffle= False)


            f1,mean_f1,kappa,bal_acc,acc = self.ft_fun(self.name,0,train_loader,test_loader,split)
            k_f1+=f1
            k_mean_f1+=mean_f1
            k_kappa+=kappa
            k_bal_acc+=bal_acc
            k_acc+=acc

            split+=1

        return k_f1/5,k_mean_f1/5,k_kappa/5,k_bal_acc/5,k_acc/5
    
    def fit(self):

        epoch_loss = 0
        scaler = torch.cuda.amp.GradScaler()
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            outputs = {'loss':[]}
            self.model.train()
            print(len(self.dataloader))
            for batch_idx,batch in tqdm(enumerate(self.dataloader)):
                with torch.cuda.amp.autocast():
                    loss,_,_,_,_= self.training_step(batch,batch_idx)

                outputs['loss'].append(loss.item())
                
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

            print(f"Pretrain Epoch {epoch}: Prev.Epoch Loss {epoch_loss:.6g} Pretrain Batch Loss:{loss.item():.6g}")
            epoch_loss = self.training_epoch_end(outputs)
            self.on_epoch_end()

            # evaluation step
            if (epoch%10==0) and (epoch!=0):
                f1,mean_f1,kappa,bal_acc,acc = self.do_kfold()

                if self.max_f1<f1:
                    chkpoint = {'eeg_model_state_dict':self.model.model.eeg_encoder.state_dict(),'best_pretrain_epoch':epoch}
                    torch.save(chkpoint, os.path.join(config.exp_path, self.name+'_best.pt'))
                    self.loggr.save(os.path.join(config.exp_path, self.name+'_best.pt'))
                    self.max_f1,self.max_kappa,self.max_bal_acc,self.max_acc = f1,kappa,bal_acc,acc

                if self.max_mean_f1<mean_f1:
                    chkpoint = {'eeg_model_state_dict':self.model.model.eeg_encoder.state_dict(),'best_pretrain_epoch':epoch}
                    torch.save(chkpoint, os.path.join(config.exp_path, self.name+'_mean_best.pt'))
                    self.loggr.save(os.path.join(config.exp_path, self.name+'_mean_best.pt'))
                    self.max_mean_f1 = mean_f1
                self.loggr.log({'F1':f1,'Mean-F1':mean_f1,'Kappa':kappa,'Bal Acc':bal_acc,'Acc':acc,'Epoch':epoch})

class sleep_ft(nn.Module):
    def __init__(self,chkpoint_pth,config,train_dl,valid_dl,pret_epoch,wandb_logger):
        super(sleep_ft,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ft_loss(chkpoint_pth,config,self.device).to(self.device)
        self.config = config
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.weight_decay = 3e-5
        self.batch_size = config.batch_size
        self.loggr = wandb_logger
        self.criterion = nn.CrossEntropyLoss()
        self.train_ft_dl = train_dl
        self.valid_ft_dl = valid_dl
        self.pret_epoch = pret_epoch
        self.max_f1 = torch.tensor(0)

        self.mean_f1 = []

        self.max_acc = torch.tensor(0)
        self.max_bal_acc = torch.tensor(0)
        self.max_kappa = torch.tensor(0)
        self.optimizer = torch.optim.Adam(self.model.parameters(),self.config.lr,betas=(self.config.beta1,self.config.beta2),weight_decay=self.weight_decay)
        self.ft_epoch = config.num_ft_epoch


    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.valid_dl

    def training_step(self,batch,batch_idx):
        data,y = batch
        data,y = data.to(self.device),y.to(self.device)
        outs = self.model(data)
        loss = self.criterion(outs,y)
        return loss

    def validation_step(self,batch,batch_idx):
        data,y = batch
        data,y = data.to(self.device),y.to(self.device)
        outs = self.model(data)
        loss = self.criterion(outs,y)
        acc = accuracy(outs,y)
        return {'loss':loss,'acc':acc,'preds':outs.detach(),'target':y.detach()} 

    def validation_epoch_end(self,outputs):
        
        epoch_preds = torch.vstack([x for x in outputs['preds']])
        epoch_targets = torch.hstack([x for x in outputs['target']])
        #epoch_loss = torch.hstack([x['loss'] for x in outputs]).mean()
        epoch_acc = torch.hstack([torch.tensor(x) for x in outputs['acc']]).mean()
        class_preds = epoch_preds.cpu().detach().argmax(dim=1)
        f1_sc = f1(epoch_preds,epoch_targets,average='macro',num_classes=5)
        kappa = cohen_kappa(epoch_preds,epoch_targets,num_classes=5)
        bal_acc = balanced_accuracy_score(epoch_targets.cpu().numpy(),class_preds.cpu().numpy())

        self.mean_f1.append(f1_sc)

        if f1_sc>self.max_f1:
            ConfusionMatrixDisplay.from_predictions(epoch_targets.cpu(),class_preds.cpu())
            self.max_f1 = f1_sc
            self.max_kappa = kappa
            self.max_bal_acc = bal_acc
            self.max_acc = epoch_acc
            self.loggr.log({f'Pretrain Epoch: Valid Confusion Matrix':plt})
            plt.close('all')

        #self.scheduler.step(epoch_loss)
    def on_train_end(self):
        self.mean_f1 = sum(self.mean_f1)/len(self.mean_f1)
        return self.max_f1,self.mean_f1,self.max_kappa,self.max_bal_acc,self.max_acc

    def fit(self):

        print("Hello")
        for ft_epoch in range(self.ft_epoch):

            # Training Loop

            self.model.train()
            ft_outputs = {'loss':[],'acc':[],'preds':[],'target':[]}
            for ft_batch_idx,ft_batch in enumerate(self.train_ft_dl):

                loss = self.training_step(ft_batch,ft_batch_idx)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Validation Loop

            self.model.eval()
            with torch.no_grad():
                for ft_batch_idx,ft_batch in enumerate(self.valid_ft_dl):
                    dct = self.validation_step(ft_batch,ft_batch_idx)
                    loss,acc,preds,target = dct['loss'],dct['acc'],dct['preds'],dct['target']
                    ft_outputs['loss'].append(loss.item())
                    ft_outputs['acc'].append(acc.item())
                    ft_outputs['preds'].append(preds)
                    ft_outputs['target'].append(target)

                self.validation_epoch_end(ft_outputs)
                print(f'FT Epoch: {ft_epoch} F1: {self.max_f1.item():.4g} Kappa: {self.max_kappa.item():.4g} B.Acc: {self.max_bal_acc.item():.4g} Acc: {self.max_acc.item():.4g}')
                #self.loggr.log({'FT Epoch':ft_epoch,'Epoch':self.pret_epoch})

        return self.on_train_end()

from genericpath import exists
from re import A, S, T
from tkinter import E
import nnet
import os
from nnet.loss import FocalLoss
import torch 
import utils
from utils import features
from utils.loader import get_loader
from torch.utils.tensorboard.writer import SummaryWriter
import tqdm
import torchaudio
import numpy as np

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

class Trainer(torch.nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args 
        self.log_folder = utils.get_log_folder(args)
        self.checkpoint_path = utils.get_ckpt_path(args)
        #clear cacher
        self.clear_cache()
        #Dataset
        self.train_loader,self.dev_loader,self.test_loader,self.noise,self.rir_loader = get_loader(args)
        #augment
        self.augment_pipeline = [
            utils.TimeDomainSpecAugment(sample_rate=16000, speeds=[100]),
            utils.TimeDomainSpecAugment(sample_rate=16000, speeds=[95, 100, 105]),
            utils.EnvCorrupt(reverb_prob=1.0, noise_prob=0.0, rir_scale_factor=1.0, reverb_loader=self.rir_loader),
            utils.EnvCorrupt(reverb_prob=0.0, noise_prob=1.0, noise_snr_low=0, noise_snr_high=15, noise_loader=self.noise),
            utils.EnvCorrupt(reverb_prob=1.0, noise_prob=1.0, noise_snr_low=0, noise_snr_high=15, rir_scale_factor=1.0, reverb_loader=self.rir_loader, 
            noise_loader=self.noise)
        ]


        self.model = nnet.get_model(args)
        self.loss_function = nnet.get_loss_function(args)
        self.optimizer = nnet.get_optimizer(self.model,args)
        self.scheduler = nnet.get_scheduler(self.optimizer,args)
        self.writer = SummaryWriter(self.log_folder)
        self.iteration = 0



    def augment_data(self,waveforms,lengths,labels):
        wave_aug_total = []
        wave_aug_total.append(waveforms)
        for layer in self.augment_pipeline:
            #print("waveforms",waveforms)
            #print("lengths",lengths)
            augmented_waveforms = layer(waveforms,lengths)
            #print("augmented_waveforms",augmented_waveforms)

            if augmented_waveforms.shape[1] > waveforms.shape[1]:
                augmented_waveforms = augmented_waveforms[:,0:waveforms.shape[1]]
            else:
                zeros_waveforms = torch.zeros_like(waveforms)
                zeros_waveforms[:,0:augmented_waveforms.shape[1]] = augmented_waveforms
                augmented_waveforms = zeros_waveforms
            wave_aug_total.append(augmented_waveforms)
        waveforms = torch.cat(wave_aug_total,dim=0)
        self.n_augmented_waveforms = len(wave_aug_total)
        lengths = torch.cat([lengths]*self.n_augmented_waveforms)
        labels = torch.cat([labels]*self.n_augmented_waveforms)
        return waveforms,lengths,labels


    
    def train_step(self,batch,batch_idx):
        x,lens,y = batch
        x = x.to(self.args.device)
        y = y.to(self.args.device)
        lens = lens.to(self.args.device)
        #print("y shape before augment",y.shape)
        x,lens,y = self.augment_data(x,lens,y)
        #rerset grad
        self.optimizer.zero_grad()
        #compute forward
        features = utils.MelSpectrogram(self.args).transform(x,lens)
        output = self.model(features,y)
        #compute loss
        #print("features shape",features.shape)
        #print(" y shape",y.shape)
        loss = self.loss_function(output,y)
        loss.backward()
        self.optimizer.step()
        return loss 




    def validation_step(self,batch,stage=None):
        with torch.no_grad():
            x,lens,y = batch
            x = x.to(self.args.device)
            y = y.to(self.args.device)
            lens = lens.to(self.args.device)

            if stage is None:
                x,lens,y = self.augment_data(x,lens,y)
                print("x",x)
                print("y",y)
                print("lens",lens)


            #inference
            features = utils.MelSpectrogram(self.args).transform(x,lens)

            #print("features",features)
            #print("x",x)
            output = self.model(features,y)
            accuracy = torch.mean((torch.argmax(output,dim=1)==y).float())
        return accuracy
    def clip_grad_norm(self):
        torch.utils.clip_grad_norm(self.model.parameters,self.args.clip_grad_norm)
    
    def load_checkpoint(self):
        self.epoch = 0
        self.accuracy = 0 
        path = self.checkpoint_path
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['mode_state_dict'])
            self.epoch = checkpoint['epoch']
            self.iteration = checkpoint['iteration']
            self.accuracy = checkpoint['accuracy']
            print(f'The latest accuracy is {self.accuracy}')
    def write_accuracy_to_tensorboard(self,epoch,accuracy,stage = 'train'):
        accuracy = float(accuracy)
        print(f"Evaluate epoch {epoch}_{stage}_accuracy: {accuracy}")
        with open(os.path.join(self.log_folder,'train_log.txt'),'a') as fin:
            fin.write(f'Evaluate epoch {epoch} {stage}_accuraacy: {accuracy}')
        metrics = {f'{stage}_accuracy': accuracy}
        self.writer.add_scalars('accuracy',metrics,epoch)
    def write_loss_to_tensorboard(self,loss):
        metrics = {'loss': loss}
        self.writer.add_scalars('loss',metrics,self.iteration)
    def limit_train_batch_hook(self,batch_index):
        if self.args.limit_train_batch_hook > 0:
            if batch_index > self.limit_train_batch_hook:
                return True
        return False
    def limit_val_batch_hook(self,batch_index):
        if self.args.limit_val_batch_hook > 0:
            if batch_index > self.limit_val_batch_hook:
                return True
        return False
    def clear_cache(self):
        if self.args.clear_cache:
            os.system(f'rm rf {self.log_folder} {self.checkpoint_path}')
    def fit(self):
        self.load_checkpoint()
        #Evaluating
        for epoch in range(self.epoch,self.args.num_epoch):
            self.model.eval()
            with tqdm.tqdm(self.test_loader,unit='it') as pbar:
                pbar.set_description(f"Evaluate epoch {epoch}")
                test_accuracy = []
                for batch_index,batch in enumerate(pbar):
                    accuracy = self.validation_step(batch,stage="test")
                    test_accuracy.append(float(accuracy))
                    pbar.set_postfix(accuracy = float(accuracy))

                    if self.limit_val_batch_hook(batch_index):
                        break 
                self.write_accuracy_to_tensorboard(epoch,accuracy,stage='test')
            if self.args.no_evaluate:
                self.model.eval()
                with tqdm.tqdm(self.dev_loader,unit='it') as pbar:
                    pbar.set_description(f'evaluate epoch {epoch}')
                    dev_accuracy = []
                    for batch_index,batch in enumerate(pbar):
                        accuracy = self.validation_step(batch,stage="Dev")
                        dev_accuracy.append(float(accuracy))
                        pbar.set_postfix(accuracy = float(accuracy))

                        if self.limit_val_batch_hook(batch_index):
                            break 
                    self.write_accuracy_to_tensorboard(epoch,accuracy,stage='dev')
            if self.args.no_evaluate:
                self.model.eval()
                with tqdm.tqdm(self.train_loader,unit='it') as pbar:
                    pbar.set_description(f'evaluate epoch {epoch}')
                    train_accuracy=[]
                    for batch_index,batch in enumerate(pbar):
                        pbar.set_description(f'evaluate epoch {epoch}')
                        train_accuracy=[]
                        for batch_index,batch in enumerate(pbar):
                            accuracy = self.validation_step(batch,stage='train')
                            train_accuracy.append(float(accuracy))
                        pbar.set_postfix(accuracy = float(accuracy))

                        if self.limit_val_batch_hook(batch_index):
                            break 
                    self.write_accuracy_to_tensorboard(epoch,accuracy,stage='train')
####################################################################################################################
            #Training
            self.model.train()
            #print("data_loader",iter(self.train_loader))
            with tqdm.tqdm(self.train_loader,unit='it') as pbar:
                pbar.set_description(f"Epoch {epoch}")
                for batch_index,batch in enumerate(pbar):
                    #print("batch_index",batch_index)
                    #print("batch",batch)
                    loss = self.train_step( batch = batch,batch_idx = batch_index)
                    pbar.set_postfix(loss=float(loss))
                    self.epoch = epoch 
                    self.iteration +=1
                    if self.iteration % self.args.log_iter == 0:
                        self.write_loss_to_tensorboard(loss)
                    if self.limit_train_batch_hook(batch_index):
                        break
            if self.accuracy < np.mean(test_accuracy):
                self.accuracy = np.mean(test_accuracy)
                self.save_checkpoint()
            if self.scheduler != None:
                self.scheduler.step(self.accuracy)




    def save_checkpoint(self):
        torch.save({'accuracy':self.accuracy,'iteration':self.iteration,'epoch':self.epoch,'model_state_dict': self.model.state_dict()},self.checkpoint_path)

    def final_fit(self):
        nnet.print_summary(self.model)
        try:
            self.fit()
        except KeyboardInterrupt:
            pass 
def train(args):
    trainer = Trainer(args)
    trainer.final_fit()


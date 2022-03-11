from torch.utils.tensorboard.writer import SummaryWriter
import numpy as np
import torch
import tqdm
import os

import util
import nnet

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

class Trainer:

    def __init__(self, args):
        # save args
        self.args = args
        self.log_folder = util.get_log_folder(args)
        self.checkpoint_path = util.get_checkpoint_path(args)
        # clear cache
        self.clear_cache()
        # get loader
        self.train_loader, self.dev_loader, self.test_loader, self.noise, self.reverb = util.get_loader(args)
        # augment data
        self.augment_pipeline = [
            util.TimeDomainSpecAugment(sample_rate=16000, speeds=[100]),
            util.TimeDomainSpecAugment(sample_rate=16000, speeds=[95, 100, 105]),
            util.EnvCorrupt(reverb_prob=1.0, noise_prob=0.0, rir_scale_factor=1.0, reverb_loader=self.reverb),
            util.EnvCorrupt(reverb_prob=0.0, noise_prob=1.0, noise_snr_low=0, noise_snr_high=15, noise_loader=self.noise),
            util.EnvCorrupt(reverb_prob=1.0, noise_prob=1.0, noise_snr_low=0, noise_snr_high=15, rir_scale_factor=1.0, reverb_loader=self.reverb, noise_loader=self.noise)
        ]
        # get model
        self.model = nnet.get_model(args)
        # get criterion
        self.criterion = nnet.get_criterion(args)
        # get optimizer
        self.optimizer = nnet.get_optimizer(self.model, args)
        # get scheduler
        self.scheduler = nnet.get_scheduler(self.optimizer, args)
        # get writer
        self.writer = SummaryWriter(self.log_folder)
        # get iteration
        self.iteration = 0
    
    def augment_data(self, wavs, lens, labels):
        
        # origin data
        wavs_aug_tot = []
        wavs_aug_tot.append(wavs)

        # augment data
        for augment in self.augment_pipeline:
            wavs_aug = augment(wavs, lens)

            # managing speed change
            if wavs_aug.shape[1] > wavs.shape[1]:
                wavs_aug = wavs_aug[:, 0 : wavs.shape[1]]
            else:
                zero_sig = torch.zeros_like(wavs)
                zero_sig[:, 0 : wavs_aug.shape[1]] = wavs_aug
                wavs_aug = zero_sig
            
            wavs_aug_tot.append(wavs_aug)

        # concat augment
        wavs = torch.cat(wavs_aug_tot, dim=0)
        self.n_augment = len(wavs_aug_tot)
        lens = torch.cat([lens] * self.n_augment)
        labels = torch.cat([labels] * self.n_augment)

        return wavs, lens, labels

    def train_step(self, batch, batch_idx):
        x, lens, y = batch
        x = x.to(self.args.device)
        lens = lens.to(self.args.device)
        y = y.to(self.args.device)
        # augment
        x, lens, y = self.augment_data(x, lens, y)
        # reset grad
        self.optimizer.zero_grad()
        # compute forward
        feats = util.MelSpectrogram(self.args).transform(x, lens)
        output = self.model(feats, y)
        # compute loss
        loss = self.criterion(output, y)
        loss.backward()
        self.optimizer.step()
        return loss

    def validation_step(self, batch, batch_idx, stage=None):
        with torch.no_grad():
            x, lens, y = batch
            x = x.to(self.args.device)
            lens = lens.to(self.args.device)
            y = y.to(self.args.device)
            if stage is None: # only augment on train set and validation set
                x, lens, y = self.augment_data(x, lens, y)
            # inference
            feats = util.MelSpectrogram(self.args).transform(x, lens)
            output = self.model(feats, y)
            accuracy = torch.mean((torch.argmax(output, dim=1) == y).float())
        return accuracy

    def clip_grad_norm(self):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)

    def load_checkpoint(self):
        self.epoch = 0
        self.accuracy = 0.
        path = self.checkpoint_path
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.epoch = checkpoint['epoch']
            self.iteration = checkpoint['iteration']
            self.accuracy = checkpoint['accuracy']
            print(f'Best accuracy: {self.accuracy}')

    def write_dev_metric_to_tensorboard(self, epoch, accuracy, stage='train'):
        accuracy = float(accuracy)
        # display
        print(f'Evaluate epoch {epoch} - {stage}_accuracy: {accuracy}')
        with open(f'{os.path.join(self.log_folder, "train_log.txt")}', 'a') as fin:
            fin.write(f'Evaluate epoch {epoch} - {stage}_accuracy: {accuracy}\n')
        # write to tensorboard
        metrics = {f'{stage}_accuracy': accuracy}
        self.writer.add_scalars('accuracy', metrics, epoch)

    def write_train_metric_to_tensorboard(self, loss):
        metrics = {'loss': float(loss)}
        self.writer.add_scalars('loss', metrics, self.iteration)

    def limit_train_batch_hook(self, batch_idx):
        if self.args.limit_train_batch > 0:
            if batch_idx > self.args.limit_train_batch:
                return True
        return False

    def limit_val_batch_hook(self, batch_idx):
        if self.args.limit_val_batch > 0:
            if batch_idx > self.args.limit_val_batch:
                return True
        return False

    def clear_cache(self):
        if self.args.clear_cache:
            os.system(f'rm -rf {self.checkpoint_path} {self.log_folder}')

    def _fit(self):
        # load checkpoint
        self.load_checkpoint()

        # training
        for epoch in range(self.epoch, self.args.num_epoch):
            ##########################################################################################
            # evalute on test
            self.model.eval()
            with tqdm.tqdm(self.test_loader, unit='it') as pbar:
                pbar.set_description(f'Evaluate epoch {epoch}')
                test_accuracy = []
                for batch_idx, batch in enumerate(pbar):
                    # validate
                    accuracy = self.validation_step(batch, batch_idx, stage='test')
                    test_accuracy.append(float(accuracy))
                    pbar.set_postfix(accuracy=float(accuracy))

                    # limit train batch hook
                    if self.limit_val_batch_hook(batch_idx):
                        break

                # print epoch summary
                self.write_dev_metric_to_tensorboard(epoch, np.mean(test_accuracy), stage='test')

            # evalute on dev + augment
            if self.args.no_evaluate:
                self.model.eval()
                with tqdm.tqdm(self.dev_loader, unit='it') as pbar:
                    pbar.set_description(f'Evaluate epoch {epoch}')
                    val_accuracy = []
                    for batch_idx, batch in enumerate(pbar):
                        # validate
                        accuracy = self.validation_step(batch, batch_idx)
                        val_accuracy.append(float(accuracy))
                        pbar.set_postfix(accuracy=float(accuracy))

                        # limit train batch hook
                        if self.limit_val_batch_hook(batch_idx):
                            break

                # print epoch summary
                self.write_dev_metric_to_tensorboard(epoch, np.mean(val_accuracy), stage='val')

            # evaluate on train + augment
            if self.args.no_evaluate:
                self.model.eval()
                with tqdm.tqdm(self.train_loader, unit='it') as pbar:
                    pbar.set_description(f'Evaluate epoch {epoch}')
                    train_accuracy = []
                    for batch_idx, batch in enumerate(pbar):
                        # validate
                        accuracy = self.validation_step(batch, batch_idx)
                        train_accuracy.append(float(accuracy))
                        pbar.set_postfix(accuracy=float(accuracy))

                        # limit train batch hook
                        if self.limit_val_batch_hook(batch_idx):
                            break

                # print epoch summary
                self.write_dev_metric_to_tensorboard(epoch, np.mean(train_accuracy), stage='train')

            ##########################################################################################
            self.model.train()
            with tqdm.tqdm(self.train_loader, unit='it') as pbar:
                pbar.set_description(f'Epoch {epoch}')
                for batch_idx, batch in enumerate(pbar):

                    # perform training step
                    loss = self.train_step(batch, batch_idx)
                    pbar.set_postfix(loss=float(loss))

                    # log
                    self.epoch = epoch
                    self.iteration += 1
                    if self.iteration % self.args.log_iter == 0:
                        self.write_train_metric_to_tensorboard(loss)

                    # limit train batch hook
                    if self.limit_train_batch_hook(batch_idx):
                        break
            
            # save checkpoint
            if self.accuracy < np.mean(test_accuracy):
                self.accuracy = np.mean(test_accuracy)
                self.save_checkpoint()
            
            # update lr via scheduler plateau
            if self.scheduler != None:
                self.scheduler.step(self.accuracy)

    def fit(self):
        nnet.print_summary(self.model)
        try:
            self._fit()
        except KeyboardInterrupt: 
            pass

    def save_checkpoint(self):
        # save checkpoint
        torch.save({
            'accuracy': self.accuracy,
            'iteration': self.iteration,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            }, self.checkpoint_path)
        print('[+] checkpoint saved')

def train(args):
    trainer = Trainer(args)
    trainer.fit()
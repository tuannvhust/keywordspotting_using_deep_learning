"""
from re import A, S
from sys import path
from torch.utils.data import Dataset,DataLoader
import os
import json
import torchaudio,torch

class KwsDataset(Dataset):
    def __init__(self,args,stage='train'):
        self.args = args 
        self.stage = stage 
        self.data , self.vocab = self.load_json()
    def load_json(self):
        data = []
        with open(f'{self.args.data_dir}/{self.stage}_{self.args.n_keyword}.json') as file:
            for line in file:
                data.append(json.loads(line))
        if self.args.n_keyword == 35:
            vocab = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go',
                     'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
                     'bed', 'bird', 'cat', 'dog', 'happy', 'house', 'marvin', 'sheila', 'tree', 'wow',
                     'backward', 'forward', 'follow', 'learn', 'visual']
        elif self.args.n_keyword == 12:
            vocab = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go',
                     'unknown', 'silence']
        else:
            raise ValueError("n_keyword must be 35 or 12")
        return data, vocab
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        audio, _ = torchaudio.load(self.data[idx]["file"])
        label = self.vocab.index(self.data[idx]['text'])
        padded_audio = padding(audio)
        length = torch.tensor(audio.shape[1]/padded_audio.shape[1])
        return padded_audio.squeeze(0),length,label





class NoiseDataset(Dataset):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.noise = self.load_json()
    def load_json(self):
        noise = []
        with open(f'{self.args.data_dir}/noise.json') as file:
            for line in file:
                noise.append(json.loads(line))
        return noise
    def __len__(self):
        return len(self.noise)
    def __getittem__(self,idx):
        noise,_ = torchaudio.load(self.noise[idx]["file"])
        padded_noise = padding(noise)
        length = torch.tensor(noise.shape[1]/padded_noise.shape[1])
        return padded_noise.squeeze(0),length





class RirDataset(Dataset):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.reverb = self.load_json()
    def load_json(self):
        reverb = []
        with open(f'{self.args.data_dir}/reverb.json') as file:
            for line in file:
                reverb.append(json.loads(line))
        return reverb
    def __len__(self):
        return len(self.reverb)
    def __getittem__(self,idx):
        reverb,_ = torchaudio.load(self.reverb[idx]["file"])
        padded_reverb = padding(reverb)
        length = torch.tensor(reverb.shape[1]/padded_reverb.shape[1])
        return padded_reverb.squeeze(0),length



def padding(x):
    if x.shape[1]>=16000:
        x = x[:,:16000]
    else:
        pad = torch.zeros(1,16000)
        pad[:,:x.shape[1]]=x
        x = pad
    return x




def get_loader(args):
    train_dataset = KwsDataset(args,stage = 'train')
    train_dataloader = DataLoader(train_dataset,shuffle = args.no_shuffle,batch_size= args.batch_size, num_workers = args.num_workers,pin_memory=args.no_pin_memory,drop_last = True)

    dev_dataset = KwsDataset(args,stage='validation')
    dev_dataloader = DataLoader(dev_dataset,shuffle = args.no_shuffle,batch_size= args.batch_size, num_workers = args.num_workers,pin_memory=args.no_pin_memory,drop_last = True)

    test_dataset = KwsDataset(args,stage='test')
    test_dataloader = DataLoader(test_dataset,shuffle = args.no_shuffle,batch_size= args.batch_size, num_workers = args.num_workers,pin_memory=args.no_pin_memory,drop_last = True)

    noise_dataset = NoiseDataset(args)
    noise_dataloader = DataLoader(noise_dataset,shuffle = args.no_shuffle,batch_size= args.batch_size, num_workers = args.num_workers,pin_memory=args.no_pin_memory,drop_last = True)

    rir_dataset = RirDataset(args)
    rir_dataloader = DataLoader(rir_dataset,shuffle = args.no_shuffle,batch_size= args.batch_size, num_workers = args.num_workers,pin_memory=args.no_pin_memory,drop_last = True)

    return train_dataloader,dev_dataloader,test_dataloader,noise_dataloader,rir_dataloader

"""
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torch
import json

class KwsDataset(Dataset):

    def __init__(self, args, stage='train'):
        self.args = args
        self.stage = stage
        self.data, self.vocab = self.load_json()

    def load_json(self):
        data = []
        with open(f'{self.args.data_dir}/{self.stage}_{self.args.n_keyword}.json') as file:
            for line in file:
                data.append(json.loads(line))

        if self.args.n_keyword == 35:
            vocab = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go',
                     'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
                     'bed', 'bird', 'cat', 'dog', 'happy', 'house', 'marvin', 'sheila', 'tree', 'wow',
                     'backward', 'forward', 'follow', 'learn', 'visual']
        elif self.args.n_keyword == 12:
            vocab = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go',
                     'unknown', 'silence']
        else:
            raise ValueError('n_keyword must be 12 or 35')

        return data, vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # raw audio and label respectively
        audio, _ = torchaudio.load(self.data[idx]['file'])
        label = self.vocab.index(self.data[idx]['text'])
        pad_audio = padding(audio)
        length = torch.tensor(audio.shape[1] / pad_audio.shape[1])
        return pad_audio.squeeze(0), length, label


class NoiseDataset(Dataset):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.noise = self.load_json()
    
    def load_json(self):
        # noise
        noise = []
        with open(f'{self.args.data_dir}/noise.json') as file:
            for line in file:
                noise.append(json.loads(line))
        return noise

    def __len__(self):
        return len(self.noise)

    def __getitem__(self, idx):
        noise, _ = torchaudio.load(self.noise[idx]['file'])
        pad_noise = padding(noise)
        length = torch.tensor(noise.shape[1] / pad_noise.shape[1])
        return pad_noise.squeeze(0), length


class RirDataset(Dataset):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.reverb = self.load_json()

    def load_json(self):
        # reverb
        reverb = []
        with open(f'{self.args.data_dir}/reverb.json') as f:
            for line in f:
                reverb.append(json.loads(line))
        return reverb

    def __len__(self):
        return len(self.reverb)

    def __getitem__(self, idx):
        reverb, _ = torchaudio.load(self.reverb[idx]['file'])
        pad_reverb = padding(reverb)
        length = torch.tensor(reverb.shape[1] / pad_reverb.shape[1])
        return pad_reverb.squeeze(0), length


def padding(x):
    if x.shape[1] >= 16000:
        x = x[:,:16000]
    else:
        pad = torch.zeros(1,16000)
        pad[:,:x.shape[1]] = x
        x = pad
    return x


def get_loader(args):
    train_dataset = KwsDataset(args, stage='train')
    train_loader = DataLoader(train_dataset,
                              shuffle=args.no_shuffle,
                              batch_size=args.batch_size,
                              num_workers=args.num_worker,
                              pin_memory=args.no_pin_memory,
                              drop_last=True)

    dev_dataset = KwsDataset(args, stage='validation')
    dev_loader = DataLoader(dev_dataset,
                            shuffle=args.no_shuffle,
                            batch_size=args.batch_size, 
                            num_workers=args.num_worker,
                            pin_memory=args.no_pin_memory,
                            drop_last=True)

    test_dataset = KwsDataset(args, stage='test')
    test_loader = DataLoader(test_dataset,
                             shuffle=args.no_shuffle,
                             batch_size=args.batch_size, 
                             num_workers=args.num_worker,
                             pin_memory=args.no_pin_memory,
                             drop_last=True)

    noise_dataset = NoiseDataset(args)
    noise_loader = DataLoader(noise_dataset,
                             shuffle=args.no_shuffle,
                             batch_size=args.batch_size, 
                             num_workers=args.num_worker,
                             pin_memory=args.no_pin_memory,
                             drop_last=True)

    rir_dataset = RirDataset(args)
    rir_loader = DataLoader(rir_dataset,
                            shuffle=args.no_shuffle,
                            batch_size=args.batch_size, 
                            num_workers=args.num_worker,
                            pin_memory=args.no_pin_memory,
                            drop_last=True)

    return train_loader, dev_loader, test_loader, noise_loader, rir_loader
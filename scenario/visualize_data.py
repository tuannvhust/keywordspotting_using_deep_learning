import torch.nn as nn
import utils , nnet
import tqdm 
import numpy as np
import os
import torch

class Visualizer(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args

    def visualize_data(self):
        path = utils.get_ckpt_path(self.args)
        if torch.cuda.is_available():
            checkpoints = torch.load(path)
        else:
            checkpoints = torch.load(path,map_location=torch.device('cpu'))
        model = nnet.get_model(self.args)
        model.load_state_dict(checkpoints['model_state_dict'])
        train_loader,_,test_loader,_,_ = utils.loader.get_loader(self.args)
        vocab = train_loader.dataset.vocab

        ###################################################
        #train
        x_train, y_train = [],[]
        with tqdm.tqdm(train_loader,unit='it') as pbar:
            for batch_index,batch in enumerate(pbar):
                x,y = self.visualize_step(model,batch)
                x_train.append(x)
                y_train.append(y)
                if self.args.limit_train_batch_hook > 0 and batch_index > self.args.limit_train_batch_hook:
                    break 
        x_train = np.concatenate(x_train)
        y_train = np.concatenate(y_train)
        #######################################################
        #test
        x_test, y_test = [],[]
        with tqdm.tqdm(test_loader,unit='it') as pbar:
            for batch_index,batch in enumerate(pbar):
                x,y = self.visualize_step(model,batch)
                x_test.append(x)
                y_test.append(y)
                if self.args.limit_val_batch_hook > 0 and batch_index > self.args.limit_val_batch_hook:
                    break 
        x_test = np.concatenate(x_test)
        y_test = np.concatenate(y_test)

        folder = utils.get_visualize_folder(self.args)
        np.savetxt(os.path.join(folder,'train','vectors.tsv'),x_train,delimiter='\t')
        with open(os.path.join(folder,'train','metadata.tsv'),'w',encoding='utf-8') as file:
            [file.write(str(vocab[int(_)])+'\n')for _ in y_train]
        np.savetxt(os.path.join(folder,'test','vectors.tsv'),x_test,delimiter='\t')
        with open(os.path.join(folder,'test','metadata.tsv'),'w',encoding='utf-8') as file:
            [file.write(str(vocab[int(_)])+'\n')for _ in y_test]           
         


    def visualize_step(self,model,batch):
        model = model.embedding 
        model = model.eval()
        with torch.no_grad():
            x,lens,y = batch 
            x = x.to(self.args.device)
            y = y.to(self.args.device)
            lens = lens.to(self.args.device)
            x = model(utils.MelSpectrogram(self.args).transform(x,lens))
            x = x.cpu().detach().numpy()
            y = y.cpu().detach().numpy()
        return x,y








def visualize_data(args):
    trainer = Visualizer(args)
    trainer.visualize_data()

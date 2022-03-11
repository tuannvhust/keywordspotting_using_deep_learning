from re import A, S
import torch.nn as nn
import torch
import utils 
import nnet
from torch.utils.tensorboard.writer import SummaryWriter





class Tester(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args 
    def final_fit(self):
        embedding_model = nnet.get_embedding_model(self.args)
        nnet.print_summary(embedding_model)
        x = torch.randn(3,101,40)
        x = embedding_model(x)
        print("x shape",x.shape)
        model = nnet.get_model(self.args)
        nnet.print_summary(model)








def test(args):
    tester = Tester(args)
    tester.final_fit()
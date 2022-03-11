from torch import nn
import torch

import nnet

def test(args):
    model = nnet.get_embedding_model(args)
    nnet.print_summary(model)
    x = torch.randn(3, 101, 40)
    x = model(x)
    print(x.shape)

    model = nnet.get_model(args)
    nnet.print_summary(model)
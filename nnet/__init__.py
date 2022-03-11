from .pooling import*
from .norm import*
from .embedding_model import*
from .conv import*
from .loss import*
from .metrics import*
import torch.nn as nn

def get_embedding_model(args):
    model = {
        'xvector': Xvector(args).to(args.device),
        'attention': Xattention(args).to(args.device),
        'bcres': BCResNet(args).to(args.device)
    }
    try:
        return model[args.embedding_model]
    except:
        NotImplementedError

def get_model(args):
    model = {
        'softmax' : SoftMax(get_embedding_model(args),args).to(args.device),
        'cosface': CosFace(get_embedding_model(args),args).to(args.device),
        'arcface': ArcFace(get_embedding_model(args),args).to(args.device),
        'sphereface': SphereFace(get_embedding_model(args),args).to(args.device),
        'adacos': AdaCos(get_embedding_model(args),args).to(args.device)
        }
    try:
        return model[args.metric]
    except:
        NotImplementedError

def get_loss_function(args):
    loss_function = {
        'ce': nn.CrossEntropyLoss(),
        'lsce': LabelSmoothingCrossEntropy(),
        'fl': FocalLoss()
    }
    try:
        return loss_function[args.loss]
    except:
        NotImplementedError
def get_optimizer(model,args):
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),lr = args.learning_rate,weight_decay=0.000002)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.Adamw(model.parameters(),lr = args.learning_rate,weight_decay=0.000002)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),lr = args.learning_rate,weight_decay=0.001,momentum = 0.9,nesterov = True)
    else:
        NotImplementedError
    return optimizer

def get_scheduler(optimizer,args):
    scheduler = {
        'none': None,
        'plateau': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max'),
    }
    return scheduler[args.scheduler]

def print_summary(model):
    print("Trainable parameters ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("none-trainable parameters", sum(p.numel() for p in model.parameters() if not p.requires_grad))
    



    

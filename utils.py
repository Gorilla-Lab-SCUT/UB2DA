import torch
import torch.optim as optim
import numpy as np
import torch.nn.functional as F


def get_optimizers(G, C, conf, F=None):
    params = []
    for key, value in dict(G.named_parameters()).items():
        if value.requires_grad and "features" in key:
            if 'bias' in key:
                params += [{'params': [value], 'lr': conf.train.multi,
                            'weight_decay': conf.train.weight_decay}]
            else:
                params += [{'params': [value], 'lr': conf.train.multi,
                            'weight_decay': conf.train.weight_decay}]
        else:
            if 'bias' in key:
                params += [{'params': [value], 'lr': 1.0,
                            'weight_decay': conf.train.weight_decay}]
            else:
                params += [{'params': [value], 'lr': 1.0,
                            'weight_decay': conf.train.weight_decay}]

    opt_g = optim.SGD(params, momentum=conf.train.sgd_momentum,
                      weight_decay=0.0005, nesterov=True)
    opt_c = optim.SGD(list(C.parameters()), lr=1.0,
                      momentum=conf.train.sgd_momentum, weight_decay=0.0005,
                      nesterov=True)
    if F is not None:
        opt_f = optim.SGD(list(F.parameters()), lr=1.0,
                        momentum=conf.train.sgd_momentum, weight_decay=0.0005,
                        nesterov=True)
        return opt_g, opt_c, opt_f
    else:
        return opt_g, opt_c
    
    
def inv_lr_scheduler(param_lr, optimizer, iter_num, gamma=10, power=0.75, init_lr=0.001, weight_decay=0.0005,
                     max_iter=10000):
    gamma = 10.0
    lr = init_lr * (1 + gamma * min(1.0, iter_num / max_iter)) ** (-power)
    i = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_lr[i]
        i += 1
    return optimizer
    
    
def entropy(p):
    p = F.softmax(p)
    return -torch.mean(torch.sum(p * torch.log(p + 1e-5), 1))


from __future__ import print_function
import torch
from apex import amp

from utils import get_optimizers, inv_lr_scheduler

def source_train(source_loader, G, C, conf):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        G.to(device)
        C.to(device)
    else:
        print('cuda is not available!!!')
        exit(0)
    batch_size = 36
    # get optimizers
    opt_g, opt_c = get_optimizers(G, C, conf)

    [G, C], [opt_g, opt_c] = amp.initialize([G, C], [opt_g, opt_c], opt_level="O1")

    G = torch.nn.DataParallel(G)
    C = torch.nn.DataParallel(C)
    param_lr_g = []
    for param_group in opt_g.param_groups:
        param_lr_g.append(param_group["lr"])
    param_lr_f = []
    for param_group in opt_c.param_groups:
        param_lr_f.append(param_group["lr"])
    # get criteria
    criterion = torch.nn.CrossEntropyLoss().cuda()

    print('train start!')
    data_iter_s = iter(source_loader)
    len_train_source = len(source_loader)
    for step in range(10000):
        G.train()
        C.train()
        if step % len_train_source == 0:
            data_iter_s = iter(source_loader)
        data_s = next(data_iter_s)
        inv_lr_scheduler(param_lr_g, opt_g, step,
                         init_lr=conf.train.lr,
                         max_iter=10000)
        inv_lr_scheduler(param_lr_f, opt_c, step,
                         init_lr=conf.train.lr,
                         max_iter=10000)
        img_s = data_s[0].cuda()
        label_s = data_s[1].cuda()

        if len(img_s) < batch_size:
            continue
        opt_g.zero_grad()
        opt_c.zero_grad()
        # Weight normalization
        C.module.weight_norm()
        # Source loss calculation
        feat = G(img_s)
        out_s = C(feat)
        loss_s = criterion(out_s, label_s)

        with amp.scale_loss(loss_s, [opt_g, opt_c]) as scaled_loss:
            scaled_loss.backward()
        opt_g.step()
        opt_c.step()
        opt_g.zero_grad()
        opt_c.zero_grad()

        if step % 100 == 0:
            print('Train [{}/{} ({:.2f}%)]\tLoss Source: {:.6f} '.format(
                step, 10000,
                100 * float(step / 10000),
                loss_s.item()))
    return G, C




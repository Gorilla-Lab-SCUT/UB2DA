from __future__ import print_function
import yaml
import easydict
import os
import torch
import torch.nn.functional as F
from apex import amp
import numpy as np
from sklearn.cluster import KMeans

from data import get_loader
from model import get_models
from utils import get_optimizers, inv_lr_scheduler, entropy
from eval import test
from train_source_model import source_train


# Training settings
import argparse

parser = argparse.ArgumentParser(description='Pytorch Universal Black-Box Domain Adaptation',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--config', type=str, default='./configs/office31-train-config.yaml',
                    help='/path/to/config/file')
parser.add_argument('--exp_name', type=str, default='office31-run1', help='experiment name')
parser.add_argument('--source_path', type=str, default='./txt/office31/source_dslr.txt', metavar='B',
                    help='path to source list')
parser.add_argument('--target_path', type=str, default='./txt/office31/target_amazon.txt', metavar='B',
                    help='path to target list')
parser.add_argument("--gpu_devices", type=int, nargs='+', default=[0], help="")

args = parser.parse_args()
config_file = args.config
conf = yaml.safe_load((open(config_file)))
conf = easydict.EasyDict(conf)
gpu_devices = ','.join([str(gpu_id) for gpu_id in args.gpu_devices])
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices

source_path = args.source_path
target_path = args.target_path
evaluation_path = args.target_path

batch_size = conf.data.dataloader.batch_size
filename_task = source_path.split("/")[-1][7:-4] + "2" + target_path.split("/")[-1][7:-4]
filename = os.path.join("record", args.exp_name, filename_task)
filename_SO = os.path.join("results_so", args.exp_name, filename_task)

if not os.path.exists(os.path.dirname(filename)):
    os.makedirs(os.path.dirname(filename))
print("record in %s " % filename)

# get data loaders
source_loader, _, _ = get_loader(source_path, source_path, source_path, batch_size=36, 
                                  return_id=True, balanced=conf.data.dataloader.class_balance)
_, target_loader, test_loader = get_loader(target_path, target_path,
                                            evaluation_path, batch_size=batch_size, return_id=True,
                                            balanced=conf.data.dataloader.class_balance)

# get numbers of shared and source classes
source_classes = set(source_loader.dataset.labels)
target_classes = set(target_loader.dataset.labels)
source_num_class = len(source_classes)
target_num_class = len(target_classes)
shared_num_class = len(source_classes & target_classes)
print('--shared_number_classes:', shared_num_class, '--source_number_classes:', source_num_class,
      '--target_number_classes:', target_num_class)
entropy_threshold = np.log(source_num_class) / 2

if not os.path.isfile(filename_SO):
    if not os.path.exists(os.path.dirname(filename_SO)):
        os.makedirs(os.path.dirname(filename_SO))
    # get source model
    G, C, _ = get_models('resnet50', num_class=source_num_class, num_cluster=conf.model.cluster_num,
                    temp=conf.model.temp)
    print('source model train begin..')
    G, C = source_train(source_loader, G, C, conf)
    print('source model train end')
    
    # modelname_SO = os.path.join("save_source_model/",
                           # args.exp_name, source_path.split("/")[-1][7:-4] + '.pt')
    # so_model_data = torch.load(modelname_SO, map_location='cpu')
    # G.load_state_dict(so_model_data['feature_extractor'])
    # C.load_state_dict(so_model_data['classifier'])
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        G.to(device)
        C.to(device)
    else:
        print('cuda is not available!!!')
        exit(0)
    prior_output = test(0, test_loader, filename, source_num_class, G, C, entropy_threshold)
    torch.save(prior_output, filename_SO)
    print('save Source Black-Box Model in %s' % filename_SO)
    print('SO++ Results: H-score: %f ; AA: %f' % (prior_output['H-score'], prior_output['AA']))
else:
    print('using the Source Black-Box Model in %s' % filename_SO)
    prior_output = torch.load(filename_SO)
    print('loading Source Black-Box Model successfully')
    print('SO++ Results: H-score: %f ; AA: %f' % (prior_output['H-score'], prior_output['AA']))
    
black_box_output = torch.from_numpy(prior_output['prob_scores']).float().cuda()

# get target model
G, C1, C2 = get_models(conf.model.base_model, num_class=source_num_class, num_cluster=conf.model.cluster_num, temp=conf.model.temp)
device = torch.device("cuda")
G.to(device)
C1.to(device)
C2.to(device)
print('Black-Box training!')

# get optimizers
opt_g, opt_c1, opt_c2 = get_optimizers(G, C1, conf, C2)

[G, C1, C2], [opt_g, opt_c1, opt_c2] = amp.initialize([G, C1, C2],
                                                      [opt_g, opt_c1, opt_c2],
                                                      opt_level="O1")

G = torch.nn.DataParallel(G)
C1 = torch.nn.DataParallel(C1)
C2 = torch.nn.DataParallel(C2)
param_lr_g = []
for param_group in opt_g.param_groups:
    param_lr_g.append(param_group["lr"])
param_lr_f = []
for param_group in opt_c1.param_groups:
    param_lr_f.append(param_group["lr"])
param_lr_f2 = []
for param_group in opt_c2.param_groups:
    param_lr_f2.append(param_group["lr"])

# initialize protopytes for neighborhood conssistency
print('initialize %d protopytes start' % conf.model.cluster_num)
G.eval()
ndata = target_loader.dataset.__len__()
target_features = np.zeros([ndata, G.module.get_feature_dim()])
for batch_idx, data in enumerate(target_loader):
    with torch.no_grad():
        img_t, index_t = data[0], data[2]
        features_t = F.normalize(G(img_t.cuda())).data.cpu().numpy()
        target_features[index_t, :] = features_t
kmeans = KMeans(n_clusters=conf.model.cluster_num, random_state=0).fit(target_features)
centers = kmeans.cluster_centers_
C2.module.weights_init_bing(torch.from_numpy(centers).float().cuda())
print('initialize %d protopytes end' % conf.model.cluster_num)


# ====== train end-to-end
def train():
    print('train start!')
    epoch = 0
    while epoch < conf.train.min_epoch:
        epoch += 1
        G.train()
        C1.train()
        C2.train()
        inv_lr_scheduler(param_lr_g, opt_g, epoch,
                             init_lr=conf.train.lr,
                             max_iter=conf.train.min_epoch)
        inv_lr_scheduler(param_lr_f, opt_c1, epoch,
                         init_lr=conf.train.lr,
                         max_iter=conf.train.min_epoch)
        inv_lr_scheduler(param_lr_f2, opt_c2, epoch,
                         init_lr=conf.train.lr,
                         max_iter=conf.train.min_epoch)
        alpha = max(1.0 - epoch / (conf.train.min_epoch), 0)
        for data_t in target_loader:
            img_t = data_t[0].cuda()
            index_t = data_t[2].cuda()

            opt_g.zero_grad()
            opt_c1.zero_grad()
            opt_c2.zero_grad()
            # Weight normalization
            C1.module.weight_norm()
            C2.module.weight_norm()

            feat_t = G(img_t)
            out_t = C1(feat_t)
            out_t_2 = C2(feat_t)

            # neighborhood consistency regularization
            after_softmax_t = F.softmax(out_t_2)
            prob_q = after_softmax_t / ((after_softmax_t.sum(0, keepdim=True) + 1e-10).pow(0.5))
            prob_q /= prob_q.sum(1, keepdim=True)
            loss_regularization = -alpha * conf.train.beta * (prob_q * torch.log(after_softmax_t + 1e-5)).sum(1).mean()

            # self-training          
            loss_pseudo = torch.tensor(0).float().cuda()

            prob_t = F.softmax(out_t, dim=1)
            _, pre_labels = torch.max(out_t, 1)
            entropy_t = -torch.sum(prob_t * torch.log(prob_t + 1e-5), dim=1)
            
            idx1 = torch.where(entropy_t < entropy_threshold - conf.train.margin)[0]
            if len(idx1) > 0:
                loss_pseudo = (1. - alpha) * conf.train.eta * entropy(out_t[idx1])
                
            idx2 = torch.where(entropy_t > entropy_threshold + conf.train.margin)[0]
            if len(idx2) > 0:
                loss_pseudo += -(1. - alpha) * conf.train.eta * entropy(out_t[idx2])
            
            # distillation
            prior_q = black_box_output[index_t]
            prior_q = prior_q / ((prior_q.sum(0, keepdim=True) + 1e-10).pow(0.5))
            prior_q /= prior_q.sum(1, keepdim=True)
            
            loss_distillation = alpha * -torch.sum(prior_q * torch.log(prob_t + 1e-5), dim=1).mean()

            all_loss = loss_pseudo + loss_regularization + loss_distillation
            with amp.scale_loss(all_loss, [opt_g, opt_c1, opt_c2]) as scaled_loss:
                scaled_loss.backward()
            opt_g.step()
            opt_c1.step()
            opt_c2.step()
            opt_g.zero_grad()
            opt_c1.zero_grad()
            opt_c2.zero_grad()
            
        if epoch % conf.train.log_interval == 0:
            print('Train [{}/{} ({:.2f}%)]\t Loss distillation: {:.6f} '
                  'Loss self-training: {:.6f} Loss regularization: {:.6f}\t'.format(
                epoch, conf.train.min_epoch,
                100 * float(epoch / conf.train.min_epoch), loss_distillation.item(), 
                loss_pseudo.item(), loss_regularization.item()))

        if epoch % conf.test.test_interval == 0:
            test(epoch, test_loader, filename, source_num_class, G, C1, entropy_threshold)


if __name__ == '__main__':
    train()

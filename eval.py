import torch
import torch.nn.functional as F
import numpy as np
import logging
from sklearn.metrics import recall_score


def test(step, dataset_test, filename, unk_class, G, C1, threshold):
    G.eval()
    C1.eval()

    all_pred = []
    all_gt = []
    entropy_scores = []
    prob_scores = np.zeros([dataset_test.dataset.__len__(), unk_class])
    for batch_idx, data in enumerate(dataset_test):
        with torch.no_grad():
            img_t, label_t, index_t = data[0], data[1], data[2]
            img_t, label_t = img_t.cuda(), label_t.cuda()
            feat = G(img_t)
            out_t = C1(feat)
            out_t = F.softmax(out_t, dim=1)
            prob_scores[index_t, :] = out_t.data.cpu().numpy()
            entr = -torch.sum(out_t * torch.log(out_t), 1).data.cpu().numpy()
            _, pred = out_t.data.max(1)
            pred = pred.cpu().numpy()

            all_gt += list(label_t.data.cpu().numpy())
            all_pred += list(pred)
            entropy_scores += list(entr)
    # list to numpy
    all_gt_np = np.array(all_gt)
    all_pred_np = np.array(all_pred)
    entropy_scores_np = np.array(entropy_scores)

    y_true = np.array(all_gt)
    y_pred = np.array(all_pred)

    pred_unk = np.where(entropy_scores_np > threshold)
    y_pred[pred_unk[0]] = unk_class

    recall_avg_auc = recall_score(y_true, y_pred, labels=np.unique(y_true), average=None)
    overall_acc = np.mean(y_true == y_pred)

    unk_idx = np.where(y_true == unk_class)  
    if len(unk_idx[0]) != 0:
        correct_unk = np.where(y_pred[unk_idx[0]] == unk_class)
        acc_unk = len(correct_unk[0]) / len(unk_idx[0])

        shared_idx = np.where(y_true != unk_class)
        shared_gt = y_true[shared_idx[0]]
        pre_shared = y_pred[shared_idx[0]]
        acc_shared = np.mean(shared_gt == pre_shared)

        h_score = 2 * acc_unk * acc_shared / (acc_unk + acc_shared)

        output = [step, list(recall_avg_auc),  
                  'AA %s' % float(recall_avg_auc.mean()),
                  'H-score %s' % float(h_score)]
                  
        # output = [step, list(recall_avg_auc), 'my_acc %s' % float(my_acc), 
                  # 'allclass per class mean acc %s' % float(recall_avg_auc.mean()),
                  # 'inclass per class mean acc %s' % float(recall_avg_auc[:-1].mean()),
                  # 'overall acc %s' % float(overall_acc), 'unknow acc %s' % float(acc_unk),
                  # 'shared acc %s' % float(acc_shared), 'H-score %s' % float(h_score), 'my_acc unkown %s' % float(my_acc_unk)]
        
        save_mat = {'gt': all_gt_np, 
                    'pred': all_pred_np,
                    'entropy_score': entropy_scores_np, 
                    'prob_scores': prob_scores,
                    'AA': recall_avg_auc.mean(),
                    'H-score': h_score,
                    'epoch': step}
    
    else:
        output = [step, list(recall_avg_auc), 'AA %s' % float(recall_avg_auc.mean()),
                  'OA %s' % float(overall_acc)]
        save_mat = {'gt': all_gt_np, 
                    'pred': all_pred_np, 
                    'entropy_score': entropy_scores_np, 
                    'prob_scores': prob_scores,
                    'AA': recall_avg_auc.mean(), 
                    'OA': overall_acc, 
                    'epoch': step}

    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=filename, format="%(message)s")
    logger.setLevel(logging.INFO)
    print('\n', output, '\n')
    logger.info(output)
    return save_mat

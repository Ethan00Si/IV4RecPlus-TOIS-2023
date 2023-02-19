from sklearn.metrics import roc_auc_score
import numpy as np

def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    '''mind.'''
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)



def hr(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_tmp = np.take(y_true, order[:k])
    return y_tmp.sum() / np.sum(y_true)
        

class judger(object):
    def __init__(self):
        super().__init__()
        self.metrics = ['hr@5','hr@10','ndcg@5','ndcg@10','mrr']

    def cal_metric(self, preds, labels):

        all_hr_5 = []
        all_hr_10 = []
        all_ndcg_5 = []
        all_ndcg_10 = []
        all_mrr = []
        all_auc = []

        for uid, uid_preds in preds.items():
            uid_labels = labels[uid]
            mrr = mrr_score(y_true=uid_labels, y_score=uid_preds)
            auc = roc_auc_score(y_true=uid_labels, y_score=uid_preds)
            all_auc.append(auc)
            ncg_5 = ndcg_score(y_true=uid_labels, y_score=uid_preds, k=5)
            ncg_10 = ndcg_score(y_true=uid_labels, y_score=uid_preds, k=10)
            hr_5 = hr(y_true=uid_labels,y_score=uid_preds,k=5)
            hr_10 = hr(y_true=uid_labels,y_score=uid_preds,k=10)

            all_ndcg_5.append(ncg_5)
            all_ndcg_10.append(ncg_10)
            all_mrr.append(mrr)
            all_hr_5.append(hr_5)
            all_hr_10.append(hr_10)
        res = dict()
        res['auc'] = np.mean(all_auc)
        res['hr@5'] = np.mean(all_hr_5)
        res['hr@10'] = np.mean(all_hr_10)
        res['ndcg@5'] = np.mean(all_ndcg_5)
        res['ndcg@10'] = np.mean(all_ndcg_10)
        res['mrr'] = np.mean(all_mrr)

        return res

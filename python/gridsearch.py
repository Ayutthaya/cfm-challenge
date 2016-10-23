import logging
import pandas
import numpy as np
import xgboost as xgb
from sklearn.metrics import matthews_corrcoef, roc_auc_score, roc_curve, auc
from sklearn.cross_validation import cross_val_score

logger = logging.getLogger('gridsearch')
hdlr = logging.FileHandler('/home/ubuntu/gridsearch.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr) 
logger.setLevel(logging.WARNING)

def mcc(tp, tn, fp, fn):
    sup = tp * tn - fp * fn
    inf = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if inf==0:
        return 0
    else:
        return sup / np.sqrt(inf)

def eval_mcc(y_true, y_prob, show=False):
    idx = np.argsort(y_prob)
    y_true_sort = y_true[idx]
    n = y_true.shape[0]
    nump = 1.0 * np.sum(y_true) # number of positive
    numn = n - nump # number of negative
    tp = nump
    tn = 0.0
    fp = numn
    fn = 0.0
    best_mcc = 0.0
    best_id = -1
    prev_proba = -1
    best_proba = -1
    mccs = np.zeros(n)
    for i in range(n):
        # all items with idx < i are predicted negative while others are predicted positive
        # only evaluate mcc when probability changes
        proba = y_prob[idx[i]]
        if proba != prev_proba:
            prev_proba = proba
            new_mcc = mcc(tp, tn, fp, fn)
            if new_mcc >= best_mcc:
                best_mcc = new_mcc
                best_id = i
                best_proba = proba
        mccs[i] = new_mcc
        if y_true_sort[i] == 1:
            tp -= 1.0
            fn += 1.0
        else:
            fp -= 1.0
            tn += 1.0
    if show:
        y_pred = (y_prob >= best_proba).astype(int)
        score = matthews_corrcoef(y_true, y_pred)
        print(score, best_mcc)
        plt.plot(mccs)
        return best_proba, best_mcc, y_pred
    else:
        return best_mcc

def roundn(y_pred, scale):
    return np.around(y_pred * scale) / scale

def mcc_eval(y_prob, dtrain):
    y_true = dtrain.get_label()
    best_mcc = eval_mcc(y_true, y_prob)
    return 'MCC', 1-best_mcc

dtrain = xgb.DMatrix('/home/ubuntu/upload/train.svm')

y=dtrain.get_label()
prior=y.sum()/len(y)

params={}
params['bst:eta'] = 0.1
params['bst:max_depth'] = 2
params['min_child_weight'] = 16
params['objective'] = 'binary:logistic'
params['nthread'] = 4
params['eval_metric'] = 'auc'
params['colsample_bytree'] = 0.7
params['learning_rate'] = 0.1
params['base_score'] = prior
params['subsample'] = 0.7

for max_depth in range(2,5):
    for learning_rate in [0.1, 0.5, 1]:
        for min_child_weight in [14, 16, 18]:
            params['bst:max_depth'] = max_depth
            params['min_child_weight'] = min_child_weight
            params['learning_rate'] = learning_rate
            logger.info('start cv with params %s' % repr(params))
            res = xgb.cv(params, dtrain, num_boost_round=40, feval=mcc_eval, verbose_eval=True, nfold=4, seed=0, early_stopping_rounds=5, show_stdv=True)
            logger.info('end of cv with params %s' % repr(params))
            logger.info('res %s' % repr(res))

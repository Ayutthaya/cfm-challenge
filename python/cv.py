# coding: utf-8

import numpy as np
import pandas
import xgboost as xgb
import sys

from configuration import *

if 'twofold' in CONFIGSTRING:
    sys.exit(0)

print('loading label')
label = pandas.read_csv(LABELFILE, sep=';')['TARGET'].values

print('loading dtrain')
dtrain = xgb.DMatrix(data=np.load(TRAINPICKLE+'.npy').T, label = label)

print('setting up params')
prior=label.mean()
params['base_score'] = prior

print('starting cross-validation')
res = xgb.cv(params, dtrain, num_boost_round=num_boost_round_cv, nfold=4, seed=0, early_stopping_rounds=20, callbacks=[xgb.callback.print_evaluation(show_stdv=True)])

print('printing results')
print(res)

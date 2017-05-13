# coding: utf-8

import numpy as np
import pandas
import xgboost as xgb

from configuration import *

print('config string: ' + CONFIGSTRING)

print('reading feature names')
with open(FEATURENAMEFILE) as featurenamefile:
    feature_names = [x.strip() for x in list(featurenamefile)]

print('loading label')
label = pandas.read_csv(LABELFILE, sep=';')
split_ID = data['ID'].max() // 2
label_fold_1 = label.ix[label['ID'] <= split_ID, 'TARGET'].values
label_fold_2 = label.ix[label['ID'] > split_ID, 'TARGET'].values

print('loading dtrain')
dtrain = xgb.DMatrix(data=np.load(TRAINPICKLE+'.npy').T, feature_names = feature_names, label = label_fold_1)

print('loading deval')
deval = xgb.DMatrix(data=np.load(TRAINPICKLE+'.npy').T, feature_names = feature_names, label = label_fold_2)

print('setting up params')
prior=label_fold_1.mean()
params['base_score'] = prior

print('xgboost params:')
for key in params:
    print(key + ': ' + repr(params[key]))

print('training model')
bst = xgb.train(params, dtrain, num_boost_round_pred, evals=(deval, 'eval'))

print('save fscore')
fscore = bst.get_fscore()
fscore_df = pandas.DataFrame(list(fscore.items()), columns=['feature', 'fscore'])
fscore_df.to_csv(FSCOREFILE)

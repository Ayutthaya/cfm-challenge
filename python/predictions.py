# coding: utf-8

import numpy as np
import pandas
import xgboost as xgb
from cooking_tools import *

from configuration import *

print('config string: ' + CONFIGSTRING)

print('reading feature names')
with open(FEATURENAMEFILE) as featurenamefile:
    feature_names = [x.strip() for x in list(featurenamefile)]

print('loading label')
label = pandas.read_csv(LABELFILE, sep=';')['TARGET'].values
if 'twofold' in CONFIGSTRING:
    label, _ = split_half_label(label)

print('loading dtrain')
dtrain = xgb.DMatrix(data=np.load(TRAINPICKLE+'.npy').T, feature_names = feature_names, label = label)

print('setting up params')
prior=label.mean()
params['base_score'] = prior

print('xgboost params:')
for key in params:
    print(key + ': ' + repr(params[key]))

print('training model')
bst = xgb.train(params, dtrain, num_boost_round_pred)

print('save fscore')
fscore = bst.get_fscore()
fscore_df = pandas.DataFrame(list(fscore.items()), columns=['feature', 'fscore'])
fscore_df.to_csv(FSCOREFILE)

print('saving model')
bst.save_model(MODELFILE)

print('loading dtest')
dtest = xgb.DMatrix(data=np.load(TESTPICKLE+'.npy').T, feature_names = feature_names)

print('computing probas')
probas = bst.predict(dtest)

print('rounding predictions')
predictions=np.round(probas).astype('int')

print('saving probas')
n_samples = probas.shape[0]
pandas.DataFrame({'ID': np.arange(1, n_samples+1), 'PROBAS': probas}).to_csv(PROBASFILE, index=False)

print('saving predictions')
pandas.DataFrame({'ID': np.arange(1, n_samples+1), 'TARGET': predictions}).to_csv(PREDICTIONSFILE, sep=';', index=False)

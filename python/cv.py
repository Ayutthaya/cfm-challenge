# coding: utf-8

import numpy as np
import pandas
import xgboost as xgb

DATADIR='/home/ubuntu/data/'
LABELFILE=DATADIR+'challenge_output_data_training_file_prediction_of_trading_activity_within_the_order_book.csv'

TRAINPICKLE='train.pkl'
TESTPICKLE='test.pkl'

RESULTSDIR='results/'
PROBASFILE=RESULTSDIR+'probas.csv'
PREDICTIONSFILE=RESULTSDIR+'predictions.csv'
MODELFILE=RESULTSDIR+'xgboost.model'

CHUNKSIZE=8
NCHUNKS=10

print('loading label')
label = pandas.read_csv(LABELFILE, sep=';', nrows=NCHUNKS)['TARGET'].values

print('loading dtrain')
dtrain = xgb.DMatrix(data=np.load(TRAINPICKLE+'.npy'), label = label)

print('setting up params')
prior=label.mean()
params={}
params['learning_rate'] = 0.1
params['bst:max_depth'] = 10
params['min_child_weight'] = 4
params['objective'] = 'binary:logistic'
params['nthread'] = 4
params['eval_metric'] = 'error'
params['lambda'] = 0.1
params['base_score'] = prior

print('starting cross-validation')
res = xgb.cv(params, dtrain, num_boost_round=1000, nfold=4, seed=0, early_stopping_rounds=100, callbacks=[xgb.callback.print_evaluation(show_stdv=True)])

print('printing results')
print(res)

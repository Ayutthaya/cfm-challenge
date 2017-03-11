# coding: utf-8

import numpy as np
import pandas
import xgboost as xgb

DATADIR='/home/ubuntu/data/'
LABELFILE=DATADIR+'challenge_output_data_training_file_prediction_of_trading_activity_within_the_order_book.csv'

TRAINPICKLE='train.pkl'
TESTPICKLE='test.pkl'
FEATURENAMEFILE='featurenamefile.txt'

RESULTSDIR='results/'
PROBASFILE=RESULTSDIR+'probas.csv'
PREDICTIONSFILE=RESULTSDIR+'predictions.csv'
MODELFILE=RESULTSDIR+'xgboost.model'
FSCOREFILE=RESULTSDIR+'fscore.csv'

CHUNKSIZE=8
NCHUNKS=10

print('reading feature names')
with open(FEATURENAMEFILE) as featurenamefile:
    feature_names = list(featurenamefile)

print('loading label')
label = pandas.read_csv(LABELFILE, sep=';')['TARGET'].values

print('loading dtrain')
dtrain = xgb.DMatrix(data=np.load(TRAINPICKLE+'.npy'), feature_names = feature_names, label = label)

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

print('training model')
num_round=300
bst = xgb.train(params, dtrain, num_round)

print('save fscore')
fscore = bst.get_fscore()
fscore = sorted(importance.items(), key=operator.itemgetter(1))
fscore_df = pandas.DataFrame(importance, columns=['feature', 'fscore'])
fscore_df.to_csv(FSCOREFILE)

print('saving model')
bst.save_model(MODELFILE)

print('loading dtest')
dtest = xgb.DMatrix(data=np.load(TESTPICKLE+'.npy'))

print('computing probas')
probas = bst.predict(dtest)

print('rounding predictions')
predictions=np.round(probas).astype('int')

print('saving probas')
n_samples = probas.shape[0]
pandas.DataFrame({'ID': np.arange(1, n_samples+1), 'PROBAS': probas}).to_csv(PROBASFILE, index=False)

print('saving predictions')
pandas.DataFrame({'ID': np.arange(1, n_samples+1), 'TARGET': predictions}).to_csv(PREDICTIONSFILE, sep=';', index=False)

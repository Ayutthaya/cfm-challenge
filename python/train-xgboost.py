# coding: utf-8

import numpy as np
import pandas
import xgboost as xgb
import gc


TRAINFILE='/home/ubuntu/data/training_input.csv'
TESTFILE='/home/ubuntu/data/testing_input.csv'
LABELFILE='/home/ubuntu/data/challenge_output_data_training_file_prediction_of_trading_activity_within_the_order_book.csv'

TRAINPICKLE='train.pkl'
TESTPICKLE='test.pkl'

MODELNAME='01.model'

CHUNKSIZE=8
NCHUNKS=10
LASTROW=7

MINUTE=8
TENMINUTES=78
HOUR=470
DAY=4679
HALFDAY=2239

USEFEATURES = ['ID',  'ask_1', 'bid_size_1', 'ask_size_1', 'bid_1', 'nb_trade', 'bid_entropy_1', 'ask_entropy_1', 'bid_size_2', 'ask_size_2', 'bid_entry_1', 'ask_entry_1', 'bid_entry_2', 'ask_entry_2']
DAY=4679
HALFDAY=2239

SKIPFEATURES = 4
NEWFEATURES = ['spread', 'bid_plus_ask', 'bid_pct']

for stage in ('train', 'test'):

    print('loading data')
    if stage=='train':
        data_all = pandas.read_csv(TRAINFILE, usecols=USEFEATURES)
    else:
        data_all = pandas.read_csv(TESTFILE, usecols=USEFEATURES)

    print('creating new features')
    data_all['spread'] = data_all['ask_1']-data_all['bid_1']
    data_all['bid_plus_ask'] = data_all['bid_size_1']-data_all['ask_size_1']
    data_all['bid_pct'] = data_all['bid_size_1']/data_all['bid_plus_ask']

    print('computing diff')
    data_diff = data_all.iloc[:, SKIPFEATURES:].diff()

    print('computing rolling mean 1 min')
    data_roll_mean_1m = data_all.iloc[7::8, SKIPFEATURES:].rolling(window=MINUTE, center=True).mean()

    print('computing rolling mean 10 min')
    data_roll_mean_10m = data_all.iloc[7::8, SKIPFEATURES:].rolling(window=TENMINUTES, center=True).mean()

    print('computing rolling mean 1 hour')
    data_roll_mean_1h = data_all.iloc[7::8, SKIPFEATURES:].rolling(window=HOUR, center=True).mean()

    print('computing rolling mean 1 day')
    data_roll_mean_1d = data_all.iloc[7::8, SKIPFEATURES:].rolling(window=DAY, center=True).mean()

    print('computing rolling std 1 min')
    data_roll_std_1m = data_all.iloc[7::8, SKIPFEATURES:].rolling(window=MINUTE, center=True).std()

    print('computing rolling std 10 min')
    data_roll_std_10m = data_all.iloc[7::8, SKIPFEATURES:].rolling(window=TENMINUTES, center=True).std()

    print('computing rolling std 1 hour')
    data_roll_std_1h = data_all.iloc[7::8, SKIPFEATURES:].rolling(window=HOUR, center=True).std()

    print('computing rolling std 1 day')
    data_roll_std_1d = data_all.iloc[7::8, SKIPFEATURES:].rolling(window=DAY, center=True).std()

    print('computing past diff')
    past_diff_feat = np.hstack([data_diff.iloc[i::8, :].values for i in range(1,8)])

    print('computing fut diff')
    fut_diff_feat = np.hstack([data_diff.iloc[i+7::8, :].values for i in range(1,9)])
    # fix end
    fut_diff_feat = np.r_[fut_diff_feat, np.zeros((1, fut_diff_feat.shape[1]))]

    print('computing pres orig')
    pres_orig_feat = data_all.iloc[7::8, SKIPFEATURES:].values

    print('computing past orig')
    past_orig_feat = np.hstack([data_all.iloc[i::8, 1:len(USEFEATURES)].values for i in range(8)])

    print('computing time feature')
    if stage=='train':
        time_feat = (data_all.iloc[7::8, 0].values-1)%DAY
    else:
        time_feat = (data_all.iloc[7::8, 0].values-1+HALFDAY)%DAY

    print('computing entropy-based features')
    entropy_feat = data_all.ix[7::8, ['bid_entropy_1', 'ask_entropy_1']]

    print('create train/testset')
    data=np.c_[past_diff_feat, fut_diff_feat, pres_orig_feat, time_feat, data_roll_mean_1m, data_roll_mean_10m, data_roll_mean_1h, data_roll_mean_1d, data_roll_std_1m, data_roll_std_10m, data_roll_std_1h, data_roll_std_1d]

    print('data shape: %s' % repr(data.shape))

    print('save train/testset')
    if stage=='train':
        np.save(TRAINPICKLE, data, allow_pickle=True)
    else:
        np.save(TESTPICKLE, data, allow_pickle=True)

#free memory
print('free memory')
gc.collect()

print('loading label')
label = pandas.read_csv(LABELFILE, sep=';')['TARGET'].values

print('loading dtrain')
dtrain = xgb.DMatrix(data=np.load(TRAINPICKLE+'.npy'), label = label)

print('setting up params')
prior=label.mean()
params={}
params['bst:eta'] = 0.1
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

print('training model')
num_round=150
bst = xgb.train(params, dtrain, num_round)

print('saving model')
bst.save_model(MODELNAME)

print('loading dtest')
dtest = xgb.DMatrix(data=np.load(TESTPICKLE+'.npy'))

print('computing predictions')
y_pred_test=np.round(bst.predict(dtest)).astype('int')

print('saving predictions')
n_samples = y_pred_test.shape[0]
pandas.DataFrame({'ID': np.arange(1, n_samples+1), 'TARGET': y_pred_test}).to_csv('/tmp/pred.csv', sep=';', index=False)

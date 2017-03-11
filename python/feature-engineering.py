# coding: utf-8

import numpy as np
import pandas

DATADIR='/home/ubuntu/data/'
TRAINFILE=DATADIR+'training_input.csv'
TESTFILE=DATADIR+'testing_input.csv'
LABELFILE=DATADIR+'challenge_output_data_training_file_prediction_of_trading_activity_within_the_order_book.csv'

TRAINPICKLE='train.pkl'
TESTPICKLE='test.pkl'
FEATURENAMEFILE='featurenamefile.txt'

CHUNKSIZE=8
NCHUNKS=10
LASTROW=7

MINUTE=8
TENMINUTES=78
HOUR=470
DAY=4679
HALFDAY=2239

USEFEATURES = ['ID', 'offset', 'ask_1', 'bid_size_1', 'ask_size_1', 'bid_1', 'nb_trade', 'bid_entropy_1', 'ask_entropy_1', 'bid_size_2', 'ask_size_2', 'bid_entry_1', 'ask_entry_1', 'bid_entry_2', 'ask_entry_2']
DAY=4679
HALFDAY=2239

SKIPIDANDOFFSET=2
SKIPFEATURES = 5
NEWFEATURES = ['spread', 'bid_plus_ask', 'bid_pct']

COMPRESSION=False

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

    if COMPRESSION:
        print('computing diff using compression')
        data_diff = data_all.ix[(data_all['offset']<=-500) | (data_all['offset']==0), SKIPFEATURES:].diff()
        n_samples_per_epoch = 3

    else:
        print('computing diff')
        data_diff = data_all.iloc[:, SKIPFEATURES:].diff()
        n_samples_per_epoch = 8

    print('computing rolling mean')
    data_roll_mean = np.hstack([data_all.iloc[7::8, SKIPFEATURES:].rolling(window=window, center=True).mean() for window in [MINUTE, TENMINUTES, HOUR, DAY]])

    print('computing rolling std')
    data_roll_std = np.hstack([data_all.iloc[7::8, SKIPFEATURES:].rolling(window=window, center=True).std() for window in [MINUTE, TENMINUTES, HOUR, DAY]])

    print('computing past diff')
    past_diff_feat = np.hstack([data_diff.iloc[i::n_samples_per_epoch, :].values for i in range(1,n_samples_per_epoch)])

    print('computing fut diff')
    fut_diff_feat= np.hstack([data_diff.iloc[i+n_samples_per_epoch-1::n_samples_per_epoch, :].values for i in range(1,n_samples_per_epoch+1)])
    # fix end
    fut_diff_feat = np.r_[fut_diff_feat, np.zeros((1, fut_diff_feat.shape[1]))]

    print('computing pres orig')
    pres_orig_feat = data_all.iloc[7::8, SKIPFEATURES:].values

    print('computing past orig')
    past_orig_feat = np.hstack([data_all.iloc[i::8, SKIPIDANDOFFSET:len(USEFEATURES)].values for i in range(8)])

    print('computing time feature')
    if stage=='train':
        time_feat = (data_all.iloc[7::8, 0].values-1)%DAY
    else:
        time_feat = (data_all.iloc[7::8, 0].values-1+HALFDAY)%DAY

    print('computing entropy-based features')
    entropy_feat = data_all.ix[7::8, ['bid_entropy_1', 'ask_entropy_1']]

    print('create train/testset')
    for data in [past_diff_feat, fut_diff_feat, pres_orig_feat, time_feat, data_roll_mean, data_roll_std]:
        print(data.shape)
    data=np.c_[past_diff_feat, fut_diff_feat, pres_orig_feat, time_feat, data_roll_mean, data_roll_std]

    print('data shape: %s' % repr(data.shape))

    BASEFEATURES = USEFEATURES[SKIPFEATURES:]
    
    past_diff_feat_names = ['past_diff_' + featurename+'_' + str(offset) for offset in range(7) for featurename in BASEFEATURES]
    future_diff_feat_names = ['fut_diff_' + featurename+'_'+ str(offset) for offset in range(8) for featurename in BASEFEATURES]
    pres_orig_feat_names = ['pres_orig_' + featurename for featurename in BASEFEATURES]
    time_feat_names = ['time']
    data_roll_mean_names = ['data_roll_mean_' + featurename + '_' + window for window in ('minute', 'tenminutes', 'hour', 'day') for featurename in BASEFEATURES]
    data_roll_std_names = ['data_roll_std_' + featurename + '_' + window for window in ('minute', 'tenminutes', 'hour', 'day') for featurename in BASEFEATURES]

    with open(FEATURENAMEFILE, 'w') as featurenamefile:
        for featurelist in (past_diff_feat_names, future_diff_feat_names, pres_orig_feat_names, time_feat_names, data_roll_std_names):
            for featurename in featurelist:
                featurenamefile.write(featurename + '\n')

    print('save train/testset')
    if stage=='train':
        np.save(TRAINPICKLE, data, allow_pickle=True)
    else:
        np.save(TESTPICKLE, data, allow_pickle=True)

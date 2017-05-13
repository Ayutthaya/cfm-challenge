# coding: utf-8

import numpy as np
import pandas

from cooking_tools import *
from configuration import *

print('loading training set')
data = pandas.read_csv(TRAINFILE)
label = pandas.read_csv(LABELFILE, sep=';')

print('split data in 2 folds')
split_ID = data['ID'].max() // 2
data_fold_1 = data[data['ID'] <= split_ID]
label_fold_1 = label.ix[label['ID'] <= split_ID, 'TARGET'].values
data_fold_2 = data[data['ID'] > split_ID]
label_fold_2 = label.ix[label['ID'] > split_ID, 'TARGET'].values

clf_nb_trade = BaggingLogisticRegression()

for stage in ('train', 'test'):

    print('loading data')
    if stage=='train':
        data = data_fold_1
        label = label_fold_1
    else:
        data = data_fold_2
        label = label_fold_2

    print('computing features')

    features = {}

    X = rolling_X(get_data(data, 'nb_trade', 0), -15, 15)

    if stage == 'train':
        clf_nb_trade.fit(X, label)

    features['nb_trade_logreg'] = clf_nb_trade.predict_proba(X)[:, 1]

    #tse = two_sided_ema(data)
    #features['two_sided_ema'] = tse

    #features['3_days_two_sided_ema'] = tse + 0.1 * (day_shift(tse, 1) + day_shift(tse, -1))

    features['consecutive_size_diff_bid_size_1'] = consecutive_diff(data, 'bid_size_1')
    features['consecutive_size_diff_ask_size_1'] = consecutive_diff(data, 'ask_size_1')

    features['simple_mmp'] = mmp(get_data(data, 'bid_size_1', 0), get_data(data, 'ask_size_1', 0))

    features['entry_based_mmp'] = mmp(get_data(data, 'bid_entry_1', 0), get_data(data, 'ask_entry_1', 0))

    for side in ('bid', 'ask'):
        for level in ('1', '2'):
            for type_ in ('size', 'entry'):
                col = '_'.join([side, type_, level])
                features[col + '_open_close_500'] = get_epoch_open_close(data, col, -500)
                features[col + '_epoch_std'] = get_epoch_std(data, col)
                features[col + '_epoch_high_low'] = get_epoch_high_low(data, col)

    features['bid_rolling_std_10'] = get_rolling(data, 'bid_1', -5, 5).std()

    features['bid_high_low_10'] = get_rolling(data, 'bid_1', -5, 5).max() - get_rolling(data, 'bid_1', -5, 5).min()

    features['bid_open_close_10'] = get_open_close(data, 'bid_1', -5, 5)

    features['bid_epoch_high_low'] = get_epoch_high_low(data, 'bid_1')

    features['bid_left_trend_7'] = np.abs(get_data(data, 'bid_1', 0) - get_rolling(data, 'bid_1', -7, 0).mean())

    features['bid_right_trend_5'] = np.abs(get_data(data, 'bid_1', 0) - get_rolling(data, 'bid_1', 0, 5).mean())

    columnlist = []
    namelist = []

    for name in features:
        namelist.append(name)
        columnlist.append(features[name])

    print('save feature names')
    with open(FEATURENAMEFILE, 'w') as featurenamefile:
        for name in namelist:
            featurenamefile.write(name + '\n')

    print('save train/testset')
    if stage=='train':
        np.save(TRAINPICKLE, np.c_[columnlist], allow_pickle=True)
    else:
        np.save(TESTPICKLE, np.c_[columnlist], allow_pickle=True)

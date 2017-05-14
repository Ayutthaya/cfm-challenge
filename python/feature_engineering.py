# coding: utf-8

import numpy as np
import pandas

from cooking_tools import *
from configuration import *


if 'twofold' in CONFIGSTRING:
    print('loading training set')
    data = pandas.read_csv(TRAINFILE)
    label = pandas.read_csv(LABELFILE, sep=';')

    print('split data in 2 folds')
    data_fold_1, data_fold_2 = split_half(data)
    label_fold_1, label_fold_2 = split_half_label(label)

for stage in ('train', 'test'):

    if 'twofold' in CONFIGSTRING:
        print('loading data')
        if stage=='train':
            data = data_fold_1
            label = label_fold_1
        else:
            data = data_fold_2
            label = label_fold_2

    else:
        print('loading data')
        if stage=='train':
            data = pandas.read_csv(TRAINFILE)
        else:
            data = pandas.read_csv(TESTFILE)

    print('computing features')

    features = {}

    tse = two_sided_ema(data)
    features['two_sided_ema'] = tse
    features['two_sided_ema_day_before'] = day_shift(tse, 1)
    features['two_sided_ema_day_next'] = day_shift(tse, -1)

    #features['3_days_two_sided_ema'] = tse + 0.1 * (day_shift(tse, 1) + day_shift(tse, -1))

    features['simple_mmp'] = mmp(get_data(data, 'bid_size_1', 0), get_data(data, 'ask_size_1', 0))

    features['entry_based_mmp'] = mmp(get_data(data, 'bid_entry_1', 0), get_data(data, 'ask_entry_1', 0))

    for side in ('bid', 'ask'):
        for level in ('1', '2'):
            for type_ in ('size', 'entry', 'entropy', 'sqentry'):
                col = '_'.join([side, type_, level])
                if type_ == 'size':
                    features[col + '_open_close_500'] = get_epoch_open_close(data, col, -500)
                if type_ == 'size' or level == '1':
                    features[col + '_epoch_high_low'] = get_epoch_high_low(data, col)
                features[col + '_epoch_std'] = get_epoch_std(data, col)
                features[col + '_consecutive_diff'] = consecutive_diff(data, col)
                features[col + '_ewm_std'] = get_data(data, col, 0).ewm(com=44).std()

    features['bid_consecutive_diff'] = consecutive_diff(data, 'bid_1')

    features['bid_rolling_std_10'] = get_rolling(data, 'bid_1', -5, 5).std()
    #features['bid_ewm_std_10'] = get_data(data, 'bid_1', 0).ewm(com=11).std()

    features['bid_high_low_10'] = get_rolling(data, 'bid_1', -5, 5).max() - get_rolling(data, 'bid_1', -5, 5).min()

    features['bid_open_close_10'] = get_open_close(data, 'bid_1', -5, 5)

    features['bid_epoch_high_low'] = get_epoch_high_low(data, 'bid_1')

    features['bid_left_trend_7'] = np.abs(get_data(data, 'bid_1', 0) - get_rolling(data, 'bid_1', -7, 0).mean())

    features['bid_right_trend_5'] = np.abs(get_data(data, 'bid_1', 0) - get_rolling(data, 'bid_1', 0, 5).mean())

    features['emp'] = get_data(data, 'bid_size_2', 0) + get_data(data, 'bid_size_1', 0) + get_data(data, 'ask_size_1', 0) + get_data(data, 'ask_size_2', 0)
    features['entry_emp'] = get_data(data, 'bid_entry_2', 0) + get_data(data, 'bid_entry_1', 0) + get_data(data, 'ask_entry_1', 0) + get_data(data, 'ask_entry_2', 0)
    features['entropy_emp'] = get_data(data, 'bid_entropy_2', 0) + get_data(data, 'bid_entropy_1', 0) + get_data(data, 'ask_entropy_1', 0) + get_data(data, 'ask_entropy_2', 0)

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

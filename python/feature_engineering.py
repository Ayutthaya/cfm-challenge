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

    features['two_sided_ema'] = two_sided_ema_nb_trade(data)
    #features['two_sided_ema_day_before'] = day_shift(tse, 1)
    #features['two_sided_ema_day_next'] = day_shift(tse, -1)
    #features['two_sided_ema_day_before_before'] = day_shift(tse, 2)
    #features['two_sided_ema_day_next_next'] = day_shift(tse, -2)

    #features['day_mean'] = nb_trade[:(length // 4679)*4679].reshape(4679, -1).mean(axis=1).repeat(length // 4679 + 1)[:length]

    #features['3_days_two_sided_ema'] = tse + 0.1 * (day_shift(tse, 1) + day_shift(tse, -1))

    features['simple_mmp'] = mmp(get_data(data, 'bid_size_1', 0), get_data(data, 'ask_size_1', 0))
    features['entry_based_mmp'] = mmp(get_data(data, 'bid_entry_1', 0), get_data(data, 'ask_entry_1', 0))

    #for type_ in ('size', 'entry', 'entropy', 'sqentry'):
    for type_ in ('size', 'entry'):
        features[type_ + '_open_close_500'] = 0
        features[type_ + '_high_low'] = 0

        for side in ('bid', 'ask'):

            #if side == 'bid':
                #sign = 1
            #else:
                #sign = -1

            for level in ('1', '2'):
                col = '_'.join([side, type_, level])
                features[type_ + '_open_close_500'] += get_epoch_open_close(data, col, -500)
                features[type_ + '_high_low'] += get_epoch_high_low(data, col)

            if level == '1':
                features[col + '_epoch_std'] = get_epoch_std(data, col)
                features[col + '_ewm_std'] = two_sided_ewm(series, 15, 'std')
                features[col + '_consecutive_diff'] = consecutive_diff(data, col)

        features[type_ + '_open_close_500'] = np.abs(features[type_ + '_open_close_500'])
        features[type_ + '_high_low'] = np.abs(features[type_ + '_high_low'])

    book_size = get_data(data, 'bid_size_1', 0) + get_data(data, 'ask_size_1', 0) + get_data(data, 'bid_size_2', 0) + get_data(data, 'ask_size_2', 0)
    features['book_size_ewm'] = two_sided_ewm(book_size, 15, 'std')

    features['bid_consecutive_diff'] = consecutive_diff(data, 'bid_1')

    #features['bid_rolling_std_10'] = get_rolling(data, 'bid_1', -5, 5).std()
    #features['bid_ewm_std_10'] = get_data(data, 'bid_1', 0).ewm(com=11).std()
    features['bid_rolling_std_10'] = two_sided_ewm(get_data(data, 'bid_1', 0), 1.5, 'std')

    features['bid_high_low_10'] = get_rolling(data, 'bid_1', -5, 5).max() - get_rolling(data, 'bid_1', -5, 5).min()
    features['bid_open_close_10'] = np.abs(get_open_close(data, 'bid_1', -5, 5))
    features['bid_epoch_high_low'] = get_epoch_high_low(data, 'bid_1')
    features['bid_left_trend_7'] = np.abs(get_data(data, 'bid_1', 0) - get_rolling(data, 'bid_1', -7, 0).mean())
    features['bid_right_trend_5'] = np.abs(get_data(data, 'bid_1', 0) - get_rolling(data, 'bid_1', 0, 5).mean())

    features['emp'] = get_data(data, 'bid_size_2', 0) + get_data(data, 'bid_size_1', 0) + get_data(data, 'ask_size_1', 0) + get_data(data, 'ask_size_2', 0)
    features['entry_emp'] = get_data(data, 'bid_entry_2', 0) + get_data(data, 'bid_entry_1', 0) + get_data(data, 'ask_entry_1', 0) + get_data(data, 'ask_entry_2', 0)
    # features['entropy_emp'] = get_data(data, 'bid_entropy_2', 0) + get_data(data, 'bid_entropy_1', 0) + get_data(data, 'ask_entropy_1', 0) + get_data(data, 'ask_entropy_2', 0)

    features['lol'] = lol(data)

    columnlist = []
    namelist = []

    #blacklist = ['bid_entry_1_epoch_high_low', 'bid_entry_2_epoch_std',
       #'bid_size_2_epoch_high_low', 'ask_entry_1_epoch_high_low',
       #'bid_open_close_10', 'bid_epoch_high_low', 'ask_entry_2_epoch_std',
       #'ask_size_2_epoch_high_low', 'bid_high_low_10',
       #'bid_sqentry_1_epoch_high_low']

    blacklist = []

    for name in features:
        if name not in blacklist:
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

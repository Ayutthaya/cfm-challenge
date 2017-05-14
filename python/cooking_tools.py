import pandas
import numpy as np


OFFSETS = (0, -10, -20, -50, -100, -200, -500, -1000)
DAY = 4679


def get_data(data, cols, offset):
    '''
    Get values for given columns and with a given offset.
    Offset can be 0, -10, -20, -50, -100, -200, -500, -1000
    '''
    return data.ix[data['offset'] == offset, cols]


def get_epoch_mean(data, column):
    '''
    Get mean for given column over an epoch.
    '''
    res = get_data(data, column, 0).values
    for offset in (-10, -20, -50, -100, -200, -500, -1000):
        res += get_data(data, column, offset).values

    return res / 8


def get_epoch_std(data, column):
    '''
    Get values for given column average over an epoch.
    '''
    return np.vstack([get_data(data, column, offset).values for offset in OFFSETS]).std(axis=0)


def get_time(data):
    '''
    Compute time
    '''
    return (data['ID'] + 2339)%4679


def get_target(path):
    '''
    Get target from output file
    '''
    return pandas.read_csv(path, sep=';')['TARGET'].values


def consecutive_diff(data, cols):
    '''
    Compute abs diff between current value and next value in time, i.e. offset == -1000 of next ID
    '''
    return np.abs(get_data(data, cols, 0).values - get_data(data, cols, -1000).shift(1).values)


def compute_accuracy(pred, target):
    assert(len(pred) == len(target))
    return (pred == target).mean()


def compute_signal_accuracy_scores(signal, target):
    assert(len(signal) == len(target))

    n_correct = (target == 1).sum()
    res = np.zeros(len(target))

    for i in signal.argsort():

        if target[i] == 0:
            n_correct += 1
        else:
            n_correct -= 1

        res[i] = float(n_correct) / len(target)

    return res[signal.argsort()]


def two_sided_ema(data):
    nb_trade = get_data(data, 'nb_trade', 0)
    com = 15

    signal_left = nb_trade.ewm(com=com).mean()   
    
    reversed_nb_trade = nb_trade.iloc[::-1]
    signal_right = reversed_nb_trade.ewm(com=com).mean()
    signal_right = signal_right.iloc[::-1]

    signal = (signal_left + signal_right) / 2
    signal.name = 'two_sided_ema'

    return signal


def mmp(q_bid, q_ask):
    return (q_bid - q_ask) / (q_bid + q_ask)


def day_shift(series, n_days):
    '''
    Shift series by n_days
    '''
    return series.shift(n_days * DAY).fillna(series)


def get_epoch_open_close(data, cols, offset):
    '''
    Get abs diff of given cols between present and offset (|open - close|)
    '''
    return np.abs(get_data(data, cols, 0).values - get_data(data, cols, offset).values)


def get_epoch_high_low(data, column):
    '''
    Get high-low for a given column over an epoch
    '''
    stack = np.vstack([get_data(data, column, offset) for offset in OFFSETS])
    high = stack.max(axis=0)
    low = stack.min(axis=0)

    return high - low


def get_rolling(data, cols, left, right):
    '''
    Get rolling window (closed edges)
    '''
    if -left == right:
        return get_data(data, cols, 0).rolling(2*right, center=True)
    if right == 0:
        assert(left < 0)
        return get_data(data, cols, 0).rolling(-left, center=False)
    if left == 0:
        assert(right > 0)
        return get_data(data, cols, 0).shift(-right + 1).rolling(right, center=False)


def get_open_close(data, cols, left, right):
    '''
    Get |open - close| in window [left, right]
    '''
    return np.abs(get_data(data, cols, 0).shift(-left) - get_data(data, cols, 0).shift(-right))


def BaggingLogisticRegression(C=0.1, n_estimators=10, max_samples=0.75, n_jobs=1):
    '''
    Create an ensemble of LogisticRegression classifiers
    '''
    from sklearn.ensemble import BaggingClassifier
    from sklearn.linear_model import LogisticRegression

    return BaggingClassifier(base_estimator=LogisticRegression(C=C), n_estimators=n_estimators, max_samples=max_samples, n_jobs=n_jobs)


def rolling_X(series, left, right):
    '''
    Create matrix X from series by rolling values
    '''
    return np.vstack([series.shift(i).fillna(series).values for i in range(-right, -left + 1)]).T


def split_half(data, cols=None):
    split_ID = data['ID'].max() // 2
    if cols is None:
        return (data[data['ID'] <= split_ID], data[data['ID'] > split_ID])
    else:
        return (data.ix[data['ID'] <= split_ID, cols], data.ix[data['ID'] > split_ID, 'TARGET'])


def split_half_label(label):
    label_1, label_2 = split_half(label, cols='TARGET')
    return (label_1.values, label_2.values)

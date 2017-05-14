import os

with open(os.path.expanduser('.configstring')) as configstringfile:
    CONFIGSTRING = list(configstringfile)[0].strip()

DATADIR='/home/ubuntu/data/'
TRAINFILE=DATADIR+'training_input.csv'
TESTFILE=DATADIR+'testing_input.csv'
LABELFILE=DATADIR+'challenge_output_data_training_file_prediction_of_trading_activity_within_the_order_book.csv'

RESULTSDIR='results/'
PROBASFILE=RESULTSDIR+'probas.csv'
PREDICTIONSFILE=RESULTSDIR+'predictions.csv'
MODELFILE=RESULTSDIR+'xgboost.model'
FEATURENAMEFILE=RESULTSDIR+'featurenamefile.txt'
FSCOREFILE=RESULTSDIR+'fscores.txt'

TRAINPICKLE='train.pkl'
TESTPICKLE='test.pkl'

params={}

if 'default' in CONFIGSTRING:
    params['learning_rate'] = 0.1
    params['bst:max_depth'] = 10
    params['min_child_weight'] = 4
    params['objective'] = 'binary:logistic'
    params['nthread'] = 4
    params['eval_metric'] = 'error'
    params['lambda'] = 0.1
    num_boost_round_cv = 200
    num_boost_round_pred = 60

elif 'slow' in CONFIGSTRING:
    params['learning_rate'] = 0.05
    params['bst:max_depth'] = 4
    params['min_child_weight'] = 6
    params['objective'] = 'binary:logistic'
    params['nthread'] = 4
    params['eval_metric'] = 'error'
    params['lambda'] = 0.1
    num_boost_round_cv = 150
    num_boost_round_pred = 142

elif 'slower' in CONFIGSTRING:
    params['learning_rate'] = 0.03
    params['bst:max_depth'] = 4
    params['min_child_weight'] = 6
    params['objective'] = 'binary:logistic'
    params['nthread'] = 4
    params['eval_metric'] = 'error'
    params['lambda'] = 0.1
    num_boost_round_cv = 600
    num_boost_round_pred = 600


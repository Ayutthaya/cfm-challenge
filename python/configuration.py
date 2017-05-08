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

params={}
params['learning_rate'] = 0.1
params['bst:max_depth'] = 10
params['min_child_weight'] = 4
params['objective'] = 'binary:logistic'
params['nthread'] = 4
params['eval_metric'] = 'error'
params['lambda'] = 0.1

num_boost_round_cv = 150
num_boost_round_pred = 75

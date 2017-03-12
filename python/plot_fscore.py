
# coding: utf-8

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
import pandas
import xgboost as xgb

MODEL = 'xgboost.model' 
FEATURES = 'featurenamefile.txt'

bst = xgb.Booster()
bst.load_model(MODEL)
fscore = bst.get_fscore()
featnames = list(open(FEATURES))
featnames = [x.strip() for x in featnames]
featdict = dict([('f'+str(i), x) for i,x in enumerate(featnames)])
df = pandas.DataFrame([(featdict[x], fscore[x]) for x in fscore], columns=['feature', 'fscore'])
df = df.sort_values(by='fscore')
df = df.reset_index(drop=True)
df[150:200].plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
plt.tight_layout()
plt.gcf().savefig('feature_importance_xgb.png')

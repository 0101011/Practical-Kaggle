import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.cross_validation import train_test_split

df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

usecols = []
for c in df_train.columns:
    if 'cont' in c:
        usecols.append(c)

x_train = df_train[usecols]
x_test = df_test[usecols]

for c in df_train.columns:
    if 'cat' in c:
        if len(df_train[c].unique()) == 2:
            uni = df_train[c].unique()[0]
            x_train[c + '_numeric'] = (df_train[c].values == uni)
            x_test[c + '_numeric'] = (df_test[c].values == uni)

y_train = df_train['loss']
id_test = df_test['id']

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=4242)

d_train = xgb.DMatrix(x_train, y_train)
d_valid = xgb.DMatrix(x_valid, y_valid)
d_test = xgb.DMatrix(x_test)

params = {}
params['eta'] = 0.0202048 # Brings back memories, doesn't it?
params['colsample_bylevel'] = 0.9
params['subsample'] = 0.9
params['silent'] = 1

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

clf = xgb.train(params, d_train, 1000, watchlist, early_stopping_rounds=20)

p_test = clf.predict(d_test)

sub = pd.DataFrame()
sub['id'] = id_test
sub['loss'] = p_test
sub.to_csv('testsub.csv', index=False)

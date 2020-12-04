# Load packages
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

from option_trading_nonprod.models.tree_based import *

#######################
# Load and prepare data
df_all = pd.read_csv('data/barchart_yf_enr_1.csv')

# Set target
df_all['reachedStrikePrice'] = np.where(df_all['maxPrice'] >= df_all['strikePrice'],1,0)

# filter set on applicable rows
# only select Call option out of the money
df = df_all[(df_all['symbolType']=='Call') & (df_all['strikePrice'] > df_all['baseLastPrice'])]

# TODO include this in preprocesseing
# clean unwanted columns
df = df.drop(columns=['Unnamed: 0','baseSymbol','symbolType','tradeTime','exportedAt','expirationDate', 'minPrice', 'maxPrice',
       'finalPrice', 'firstPrice'])

########################
# Split in train and test
X = df.drop(columns='reachedStrikePrice')
y = df['reachedStrikePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

#####################
# Train and predict

# Random Forest
from sklearn.ensemble import RandomForestClassifier

# Random Forest classifier model
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_train,y_train)

# Catboost
params = {'iterations':300}

getwd = os.getcwd()
cb_model = fit_cb(X_train, y_train, X_val, y_val, params, save_model = True, cb_path=getwd+'/trained_models/', name='cb_model_v1')

# Choose model
model = cb_model

# Make predictions
prob = model.predict_proba(X_test)[:,1]

pred_df = pd.DataFrame({'prob':prob, 'actual':y_test})
pred_df['pred'] = np.where(pred_df['prob'] >= 0.5,1,0)

#####################
# Measure performance
from option_trading_nonprod.validation.classification import showConfusionMatrix, plotCurveAUC

plotCurveAUC(pred_df['prob'],pred_df['actual'],type='roc')
showConfusionMatrix(pred_df['pred'], actual=pred_df['actual'])

#####################
# Visualize

######################
# Test out predictions
# profitability
df_test = df_all.loc[pred_df.index,:]
df_test['prob'] =  pred_df['prob']
df_test['maxProfit'] = df_test['maxPrice'] - df_test['baseLastPrice']
df_test['aimedProfit'] = np.where(df_test['maxPrice'] >= df_test['strikePrice'],df_test['strikePrice'], df_test['finalPrice']) - df_test['baseLastPrice']
df_test['priceDiffPerc'] = df_test['strikePrice'] / df_test['baseLastPrice']
# Select based on parameters
# Subsetting the predictions
threshold = 0.5
maxBasePrice = 200
minDaysToExp = 3
maxDaysToExp = 20
minStrikeIncrease = 1.05

df_test = df_test[(df_test['prob'] > threshold) &
    (df_test['symbolType']=='Call') &
    (df_test['daysToExpiration'] < maxDaysToExp) &
    (df_test['priceDiffPerc'] > minStrikeIncrease) &
    (df_test['daysToExpiration'] > minDaysToExp) &
    (df_test['baseLastPrice'] < maxBasePrice)]

df_test.describe()


######################
# Tune hyperparameters
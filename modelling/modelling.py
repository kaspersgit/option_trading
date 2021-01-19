# Load packages
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

from option_trading_nonprod.models.tree_based import *
from option_trading_nonprod.validation.calibration import *
from option_trading_nonprod.process.stock_price_enriching import *

#######################
# Load and prepare data
df_all = pd.read_csv('data/barchart_yf_enr_1.csv')

# clean out duplicates to be sure
df_all = df_all.drop(axis=1, columns='Unnamed: 0')
df_all = df_all.drop_duplicates(subset=['baseSymbol','symbolType','strikePrice','expirationDate','exportedAt'])

# Add internal (within same batch) information
df_all = enrich_df(df_all)

# Set target
df_all['reachedStrikePrice'] = np.where(df_all['maxPrice'] >= df_all['strikePrice'],1,0)

# filter set on applicable rows
# only select Call option out of the money
df = df_all[(df_all['symbolType']=='Call') & (df_all['strikePrice'] > df_all['baseLastPrice'])]

# feature selection
features = ['reachedStrikePrice',
            'openInterestCall',
            'meanStrikeCallPerc',
            'volatility',
            'baseLastPrice',
            'strikePrice',
            'askPrice',
            'priceDiffPerc',
            'openInterest',
            'volume',
            'openInterestCumSum',
            'meanStrikePut',
            'midpointPerc',
            'volumeCall']

# TODO include this in preprocesseing
# clean unwanted columns
df = df.drop(columns=['baseSymbol','symbolType','tradeTime','exportedAt','expirationDate', 'minPrice', 'maxPrice',
       'finalPrice', 'firstPrice'])

# df = df[features]

########################
# Split in train and test
X = df.drop(columns='reachedStrikePrice')
y = df['reachedStrikePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

#####################
# Train and predict
# AdaBoost classifier
# either DEV(ELOPMENT) or PROD(UCTION)
# v1x0 trained on data with max expirationDate 2020-10-30
# v1x1 trained on data with max expirationDate 2020-12-18

train_type = 'DEV'
version = 'v1x1'
if train_type == 'DEV':
    X_fit = X_train
    y_fit = y_train
    df_test = df_all.loc[X_test.index,:]
    df_test.to_csv("validation/test_df.csv")
elif train_type == 'PROD':
    X_fit = pd.concat([X_train, X_test])
    y_fit = pd.concat([y_train, y_test])

getwd = os.getcwd()
params = {'n_estimators':1000, 'learning_rate':0.5, 'random_state':42}
AB_model = fit_AdaBoost(X_fit, y_fit, X_val, y_val, params, save_model = False, ab_path=getwd+'/trained_models/', name='AB64_'+version)
# Calibrate pre trained model
Cal_AB_model = calibrate_model(AB_model, X_val, y_val, method='sigmoid', save_model=True, path=getwd+'/trained_models/', name=train_type+'_c_AB64_'+version)

params = {'n_estimators':1000, 'learning_rate':0.5, 'random_state':42}
GBC_model = fit_GBclf(X_train, y_train, X_val, y_val, params, save_model = True, gbc_path=getwd+'/trained_models/', name='GB64_'+version)

Adaprob = AB_model.predict_proba(X_val)[:,1]
GBprob = GBC_model.predict_proba(X_val)[:,1]

# test Dataset
prob = Cal_AB_model.predict_proba(X_test)[:,1]

###########
# Choose model
# Load model
getwd = os.getcwd()
with open(getwd+'/trained_models/AB_v1.sav', 'rb') as file:
    model = pickle.load(file)
with open(getwd+'/trained_models/DEV_c_AB64_v2.sav', 'rb') as file:
    calib_model = pickle.load(file)

model = Cal_AB_model

# Make predictions
prob = model.predict_proba(X_test[model.feature_names])[:,1]

pred_df = pd.DataFrame({'prob':prob, 'actual':y_test})
pred_df['pred'] = np.where(pred_df['prob'] >= 0.5,1,0)

#####################
# Measure performance
from option_trading_nonprod.validation.classification import showConfusionMatrix, plotCurveAUC

# AUC
plotCurveAUC(pred_df['prob'],pred_df['actual'], title='all data', type='roc')

# Confucion matrix
showConfusionMatrix(pred_df['pred'], actual=pred_df['actual'])

# Calibration plot
plotCalibrationCurve(pred_df['actual'], pred_df['prob'], bins=10)

#####################
# Visualize


######################
# Test out predictions
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)
# profitability
df_test = df_all.loc[pred_df.index,:]
df_test['prob'] =  pred_df['prob']
df_test['priceDiffPerc'] = df_test['strikePrice'] / df_test['baseLastPrice']
df_test['maxProfit'] = df_test['maxPrice'] - df_test['baseLastPrice']
df_test['aimedProfit'] = np.where(df_test['maxPrice'] >= df_test['strikePrice'],df_test['strikePrice'], df_test['finalPrice']) - df_test['baseLastPrice']
# Select based on parameters
# Subsetting the predictions
threshold = 0.5
maxBasePrice = 200
minDaysToExp = 3
maxDaysToExp = 20
minStrikeIncrease = 1.05

df_select = df_test[
    (df_test['prob'] > threshold) &
    (df_test['symbolType']=='Call') &
    (df_test['daysToExpiration'] < maxDaysToExp) &
    (df_test['priceDiffPerc'] > minStrikeIncrease) &
    (df_test['daysToExpiration'] > minDaysToExp) &
    (df_test['baseLastPrice'] < maxBasePrice)
]

df_select.describe()
plotCurveAUC(df_select['prob'],df_select['actual'],type='roc')

# Subset based on highest profit (%)
df_test['profitPerc'] = df_test['aimedProfit'] / df_test['baseLastPrice']
df_highest = df_test.sort_values('profitPerc', ascending=False).head(400)
df_highest['profTimesProb'] = df_highest['priceDiffPerc'] * df_highest['prob']
df_highest[['baseSymbol','baseLastPrice','strikePrice','priceDiffPerc','maxPrice','aimedProfit','profitPerc','prob','profTimesProb','reachedStrikePrice']].head(50)

# calibration plot
plotCalibrationCurve(df_highest['reachedStrikePrice'], df_highest['prob'], bins=10)

######################
# Tune hyperparameters
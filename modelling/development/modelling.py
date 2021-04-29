# Load packages
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GroupShuffleSplit

from option_trading_nonprod.models.calibrate import *
from option_trading_nonprod.models.tree_based import *
from option_trading_nonprod.validation.calibration import *
from option_trading_nonprod.process.stock_price_enriching import *

#######################
# Load and prepare data
df_all = pd.read_csv('data/barchart_yf_enr_1x2.csv')

# clean out duplicates to be sure
df_all = df_all.drop_duplicates(subset=['baseSymbol','symbolType','strikePrice','expirationDate','exportedAt'])

# Add internal (within same batch) information
df_all = batch_enrich_df(df_all, groupByColumns=['exportedAt', 'baseSymbol', 'symbolType', 'inTheMoney'])

# Set target
df_all['reachedStrikePrice'] = np.where(df_all['maxPrice'] >= df_all['strikePrice'],1,0)

# filter set on applicable rows
# only select Call option out of the money
df = df_all[(df_all['symbolType'] == 'Call') & (df_all['strikePrice'] > df_all['baseLastPrice'])]

# feature selection
features_all = ['strikePrice'
    , 'daysToExpiration'
    , 'bidPrice'
    , 'midpoint'
    , 'askPrice'
    , 'lastPrice'
    , 'openInterest'
    , 'volumeOpenInterestRatio'
    , 'volatility'
    , 'open'
    , 'high'
    , 'low'
    , 'close'
    , 'adj_close'
    , 'volume'
    , 'MACD_2_4_9'
    , 'MACDh_2_4_9'
    , 'MACDs_2_4_9'
    , 'RSI_14'
    , 'OBV'
    , 'BBL_5_2.0'
    , 'BBM_5_2.0'
    , 'BBU_5_2.0'
    , 'BBB_5_2.0'
    , 'priceDiff'
    , 'priceDiffPerc'
    , 'inTheMoney'
    , 'nrOptions'
    , 'strikePriceCum'
    , 'volumeTimesStrike'
    , 'nrCalls'
    , 'meanStrikeCall'
    , 'sumVolumeCall'
    , 'sumOpenInterestCall'
    , 'sumVolumeTimesStrikeCall'
    , 'weightedStrikeCall'
    , 'nrPuts'
    , 'meanStrikePut'
    , 'sumVolumePut'
    , 'sumOpenInterestPut'
    , 'sumVolumeTimesStrikePut'
    , 'weightedStrikePut'
    , 'volumeCumSum'
    , 'openInterestCumSum'
    , 'nrHigherOptions'
    , 'higherStrikePriceCum'
    , 'meanStrikeCallPerc'
    , 'meanStrikePutPerc'
    , 'midpointPerc'
    , 'meanHigherStrike']

features = ['baseLastPrice'
    , 'strikePrice'
    , 'daysToExpiration'
    , 'bidPrice'
    , 'midpoint'
    , 'askPrice'
    , 'lastPrice'
    , 'volume'
    , 'openInterest'
    , 'volumeOpenInterestRatio'
    , 'volatility'
    , 'priceDiff'
    , 'priceDiffPerc'
    , 'inTheMoney'
    , 'nrOptions'
    , 'strikePriceCum'
    , 'volumeTimesStrike'
    , 'nrCalls'
    , 'meanStrikeCall'
    , 'sumVolumeCall'
    , 'sumOpenInterestCall'
    , 'sumVolumeTimesStrikeCall'
    , 'weightedStrikeCall'
    , 'nrPuts'
    , 'meanStrikePut'
    , 'sumVolumePut'
    , 'sumOpenInterestPut'
    , 'sumVolumeTimesStrikePut'
    , 'weightedStrikePut'
    , 'volumeCumSum'
    , 'openInterestCumSum'
    , 'nrHigherOptions'
    , 'higherStrikePriceCum'
    , 'meanStrikeCallPerc'
    , 'meanStrikePutPerc'
    , 'midpointPerc'
    , 'meanHigherStrike']


########################
# Split in train, validation, test and out of time
# target  used
target = 'reachedStrikePrice'

# Split in train, validation, test and out of time
# Take most recent observations for out of time set (apprx last 5000 observations)
exportDateLast5000 = df.iloc[-5000]['exportedAt']
df_oot = df[df['exportedAt'] >= exportDateLast5000]
df_rest = df.drop(df_oot.index, axis=0).reset_index(drop=True)

# test to split keeping exportedAt column always in same group
gss = GroupShuffleSplit(n_splits=1, train_size=.8, random_state=42)
gss.get_n_splits()

# split off test set
test_groupsplit = gss.split(df_rest, groups = df_rest['exportedAt'])
train_idx, test_idx = next(test_groupsplit)
df_rest2 = df_rest.loc[train_idx]
df_test = df_rest.loc[test_idx]

# split off validation set
df_rest2 = df_rest2.reset_index(drop=True)
val_groupsplit = gss.split(df_rest2, groups = df_rest2['exportedAt'])
train_idx, val_idx = next(val_groupsplit)
df_train = df_rest2.loc[train_idx]
df_val = df_rest2.loc[val_idx]

# clean unwanted columns for model training
X_train = df_train.drop(columns=[target])
y_train = df_train[target]

X_val = df_val.drop(columns=[target])
y_val = df_val[target]

X_test = df_test.drop(columns=[target])
y_test = df_test[target]

X_oot = df_oot.drop(columns=[target])
y_oot = df_oot[target]

print("Train shape: {}\nValidation shape: {}\nTest shape: {}\nOut of time shape: {}".format(X_train.shape,X_val.shape,X_test.shape,X_oot.shape))
#####################
# Train and predict
# AdaBoost classifier
# either DEV(ELOPMENT) or PROD(UCTION)
# v1x0 trained on data with max expirationDate 2020-10-30
# v1x1 trained on data with max expirationDate 2020-12-18

train_type = 'DEV'
version = 'v4x3'
if train_type == 'DEV':
    X_fit = X_train
    y_fit = y_train
    df_test = df_all.loc[X_test.index,:]
    df_test.to_csv("data/validation/test_df.csv")
    df_oot.to_csv("data/validation/oot_df.csv")
elif train_type == 'PROD':
    X_fit = pd.concat([X_train, X_test])
    y_fit = pd.concat([y_train, y_test])

getwd = os.getcwd()
params = {'n_estimators':1000, 'learning_rate':0.5, 'random_state':42}
AB_model = fit_AdaBoost(X_fit, y_fit, X_val, y_val, params, save_model = False, ab_path=getwd+'/trained_models/', name='AB64_'+version)
# Calibrate pre trained model
Cal_AB_model = calibrate_model(AB_model, X_val, y_val, method='sigmoid', save_model=True, path=getwd+'/trained_models/', name=train_type+'_c_AB64_'+version)

# Fillna with 0 (missing indicators due to short history)
X_train.fillna(0, inplace=True)
X_val.fillna(0, inplace=True)
params = {'n_estimators':1000, 'learning_rate': 0.01, 'min_samples_split':8, 'max_depth':4, 'random_state':42, 'subsample':1}
GBC_model = fit_GBclf(X_train[features], y_train, X_val[features], y_val, params, save_model = True, gbc_path=getwd+'/trained_models/', name='GB64_'+version)
# Calibrate pre trained model
Cal_GB_model = calibrate_model(GBC_model, X_val, y_val, method='sigmoid', save_model=True, path=getwd+'/trained_models/', name=train_type+'_c_GB64_'+version)

###########
# Choose model
# Load model
getwd = os.getcwd()
with open(getwd+'/trained_models/DEV_c_GB64_v4x3.sav', 'rb') as file:
    gb_model = pickle.load(file)
with open(getwd+'/trained_models/DEV_c_AB64_v1x3.sav', 'rb') as file:
    ab_model = pickle.load(file)

model = gb_model

# Make predictions
prob = model.predict_proba(X_test[model.feature_names])[:,1]

pred_df = pd.DataFrame({'prob': prob, 'actual': y_test})
pred_df['pred'] = np.where(pred_df['prob'] >= 0.5, 1, 0)
df_test['prob'] = prob

#####################
# Measure performance
from option_trading_nonprod.validation.classification import showConfusionMatrix, plotCurveAUC

# AUC
plotCurveAUC(pred_df['prob'],pred_df['actual'], title='all data', type='roc')

# Confucion matrix
showConfusionMatrix(pred_df['pred'], actual=pred_df['actual'])

# Calibration plot
plotCalibrationCurve(pred_df['actual'], pred_df['prob'], title='all data', bins=10)

# Load packages
import pandas as pd
import numpy as np
import os
os.chdir('/home/pi/Documents/python_scripts/option_trading')
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from option_trading_nonprod.models.tree_based import *
from option_trading_nonprod.models.calibrate import *
from option_trading_nonprod.process.stock_price_enriching import *



#######################
# Load and prepare data
df_all = pd.read_csv('data/barchart_yf_enr_1.csv')

# Set target
df_all['reachedStrikePrice'] = np.where(df_all['maxPrice'] >= df_all['strikePrice'],1,0)

# batch enrich dataset
df_all = batch_enrich_df(df_all)

# filter set on applicable rows
# only select Call option out of the money
df = df_all[(df_all['symbolType']=='Call') & (df_all['strikePrice'] > df_all['baseLastPrice'])]


# clean unwanted columns
df = df.drop(columns=['Unnamed: 0','baseSymbol','symbolType','tradeTime','exportedAt','expirationDate', 'minPrice', 'maxPrice',
       'finalPrice', 'firstPrice'])

print('Total train data shape: {}'.format(df.shape))
print('Minimum strike price increase: {}'.format(round((df['strikePrice'] / df['baseLastPrice']).min(), 2)))
print('Maximum strike price increase: {}'.format(round((df['strikePrice'] / df['baseLastPrice']).max(), 2)))
print('Minimum nr days until expiration: {}'.format(df['daysToExpiration'].min()))
print('Maximum nr days until expiration: {}'.format(df['daysToExpiration'].max()))

target = 'reachedStrikePrice'

features = ['priceDiffPerc'
    , 'midpointPerc'
    , 'volatility'
    , 'priceDiff'
    , 'meanStrikeCallPerc'
    , 'baseLastPrice'
    , 'daysToExpiration'
    , 'sumVolumeCall'
    , 'sumVolumeTimesStrikeCall'
    , 'sumOpenInterestCall'
    , 'volumeOpenInterestRatio'
    , 'meanStrikePut'
    , 'openInterest'
    , 'meanStrikePutPerc'
    , 'volumeCumSum'
    , 'volumeTimesStrike'
    , 'volume'
    , 'openInterestCumSum'
    , 'weightedStrikeCall'
    , 'lastPrice']

print('Nr of features included: {}'.format(len(features)))

df_rest = df[features]
########################
# Split in train and test
# test to split keeping exportedAt column always in same group
gss = GroupShuffleSplit(n_splits=1, train_size=.75, random_state=42)
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

#####################
# Train
# Classifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

train_type = 'PROD'
version = 'v1x3'
algorithm = 'AB'
if train_type == 'DEV':
    X_fit = X_train
    y_fit = y_train
    df_test = df_all.loc[X_test.index,:]
    df_test.to_csv("validation/test_df.csv")
elif train_type == 'PROD':
    X_fit = pd.concat([X_train, X_test])
    y_fit = pd.concat([y_train, y_test])

print('Train data shape: {}'.format(X_fit.shape))
print('Calibration data shape: {}'.format(X_val.shape))
print('Train type: {}\nVersion: {}\nAlgorithm: {}'.format(train_type, version, algorithm))
print('Training uncalibrated model...')

getwd = os.getcwd()
if algorithm == 'AB':
    params = {'n_estimators':1000, 'learning_rate':0.5, 'random_state':42}
    uncal_model = fit_AdaBoost(X_fit, y_fit, X_val, y_val, params, save_model = False, ab_path=getwd+'/trained_models/', name='{}_{}32_{}'.format(train_type, algorithm, version))
elif algorithm == 'GB':
    params = {'n_estimators': 3000, 'learning_rate': 0.05, 'min_samples_split': 8, 'subsample': 0.8, 'max_depth': 4, 'random_state':42}
    uncal_model = fit_GBclf(X_fit, y_fit, X_val, y_val, params, save_model = False, gbc_path=getwd+'/trained_models/', name='{}_{}32_{}'.format(train_type, algorithm, version))

print('Training uncalibrated model... Done!')

print('Calibrate and save model...')
# calibrate and save classifier
cal_model = calibrate_model(uncal_model, X_val, y_val, method='sigmoid', save_model=True, path=getwd+'/trained_models/', name='{}_c_{}32_{}'.format(train_type, algorithm, version))


print('Calibrate and save model... Done!')

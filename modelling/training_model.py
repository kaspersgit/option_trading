# Load packages
import pandas as pd
import numpy as np
import os
os.chdir('/home/pi/Documents/python_scripts/option_trading')
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import auc, roc_curve
from option_trading_nonprod.aws import *
from option_trading_nonprod.models.tree_based import *
from option_trading_nonprod.models.calibrate import *
from option_trading_nonprod.process.stock_price_enriching import *


###### import data from S3
# Set source and target for bucket and keys
today = '2021-04-15'
source_bucket = 'project-option-trading-output'
source_key = 'train_data/barchart/enriched_on_{}.csv'.format(today)

# print status of variables
print('Source bucket: {}'.format(source_bucket))
print('Source key: {}'.format(source_key))

# import data
if platform.system() == 'Darwin':
    s3_profile = 'mrOption'
else:
    s3_profile = 'default'

df_all = load_from_s3(profile=s3_profile, bucket=source_bucket, key_prefix=source_key)
print("Raw imported data shape: {}".format(df_all.shape))

# Set target
df_all['reachedStrikePrice'] = np.where(df_all['maxPrice'] >= df_all['strikePrice'],1,0)

# batch enrich dataset
df_all = batch_enrich_df(df_all)

# filter set on applicable rows
# only select Call option out of the money
df = df_all[(df_all['symbolType']=='Call') & (df_all['strikePrice'] > 1.05 * df_all['baseLastPrice'])]
df = df.reset_index(drop=True)

print('Null values: \n{}'.format(df.isna().sum()))
df = df[~ df['maxPrice'].isna()].reset_index(drop=True)
# df.fillna(0, inplace=True)

print('Total train data shape: {}'.format(df.shape))
print('Minimum strike price increase: {}'.format(round((df['strikePrice'] / df['baseLastPrice']).min(), 2)))
print('Maximum strike price increase: {}'.format(round((df['strikePrice'] / df['baseLastPrice']).max(), 2)))
print('Minimum nr days until expiration: {}'.format(df['daysToExpiration'].min()))
print('Maximum nr days until expiration: {}'.format(df['daysToExpiration'].max()))

target = 'reachedStrikePrice'

features = ['baseLastPrice', 'strikePrice', 'daysToExpiration', 'bidPrice', 'midpoint',
                 'askPrice', 'lastPrice', 'volume', 'openInterest',
                 'volumeOpenInterestRatio', 'volatility']

features_ext = ['midpointPerc', 'priceDiff', 'priceDiffPerc', 'bidPrice', 'strikePriceCum', 'midpoint', 'lastPrice', 'weightedStrikeCall', 'strikePrice',
            'meanStrikeCall', 'sumVolumeTimesStrikeCall', 'askPrice', 'meanStrikePut', 'daysToExpiration', 'meanStrikeCallPerc',
            'sumVolumeTimesStrikePut', 'weightedStrikePut', 'volumeTimesStrike', 'OBV', 'BBU_5_2.0', 'BBB_5_2.0', 'volatility', 'BBM_5_2.0',
            'sumOpenInterestCall', 'BBL_5_2.0', 'MACD_2_20_9', 'sumOpenInterestPut', 'volumeCumSum', 'MACDs_2_20_9', 'higherStrikePriceCum']

print('Nr of features included: {}'.format(len(features)))

########################
# Split in train and test
# test to split keeping exportedAt column always in same group
gss = GroupShuffleSplit(n_splits=1, train_size=.75, random_state=42)
gss.get_n_splits()

# split off test set
test_groupsplit = gss.split(df, groups = df['exportedAt'])
train_idx, test_idx = next(test_groupsplit)
df2 = df.loc[train_idx]
df_test = df.loc[test_idx]

# split off validation set
df2 = df2.reset_index(drop=True)
val_groupsplit = gss.split(df2, groups = df2['exportedAt'])
train_idx, val_idx = next(val_groupsplit)
df_train = df2.loc[train_idx]
df_val = df2.loc[val_idx]

# clean unwanted columns for model training
X_train = df_train[features]
y_train = df_train[target]

X_val = df_val[features]
y_val = df_val[target]

X_test = df_test[features]
y_test = df_test[target]

#####################
# Train
# Classifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

train_type = 'PROD'
version = 'v1x3'
algorithm = 'GB'
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
    params = {'n_estimators':3000, 'max_depth': 10, 'max_features': 8, 'min_samples_split': 225, 'subsample': 0.808392563444737, 'learning_rate': 0.00010030663168798627}
    uncal_model = fit_GBclf(X_fit, y_fit, X_val, y_val, params, save_model = False, gbc_path=getwd+'/trained_models/', name='{}_{}32_{}'.format(train_type, algorithm, version))

print('Training uncalibrated model... Done!')

xVar, yVar, thresholds = roc_curve(y_val, uncal_model.predict_proba(X_val)[:, 1])
roc_auc = auc(xVar, yVar)
print('Model has a ROC on validation set of: {}'.format(roc_auc))

print('Calibrate and save model...')
# calibrate and save classifier
cal_model = calibrate_model(uncal_model, X_val, y_val, method='sigmoid', save_model=True, path=getwd+'/trained_models/', name='{}_c_{}32_{}'.format(train_type, algorithm, version))


print('Calibrate and save model... Done!')

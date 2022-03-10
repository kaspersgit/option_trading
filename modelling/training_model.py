# Load packages
import pandas as pd
import numpy as np
import os, sys, json
os.chdir('/home/pi/Documents/python_scripts/option_trading')
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import auc, roc_curve
from option_trading_nonprod.aws import *
from option_trading_nonprod.models.tree_based import *
from option_trading_nonprod.models.calibrate import *
from option_trading_nonprod.process.train_modifications import *
from option_trading_nonprod.process.pre_train import *
from option_trading_nonprod.process.stock_price_enriching import *
from option_trading_nonprod.other.trading_strategies import *

# 32 or 64 bit system
n_bits = 32 << bool(sys.maxsize >> 32)

###### import data from S3
# Set source and target for bucket and keys
source_bucket = 'project-option-trading-output'
source_key = 'enriched_data/barchart/expired_on_'

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
with open('other_files/config_file.json') as json_file:
    config = json.load(json_file)

hprob_config = config['high_probability']
hprof_config = config['high_profitability']
included_options = config['included_options']

# Filter on basics (like days to expiration and contract type)
df = dfFilterOnGivenSetOptions(df_all, included_options)

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

features = ['baseLastPrice',
            'strikePrice',
            'daysToExpiration',
            'bidPrice',
            'midpoint',
            'askPrice',
            'lastPrice',
            'openInterest',
            'volumeOpenInterestRatio',
            'volatility',
            'volume',
            'nrOptions',
            'strikePriceCum',
            'volumeTimesStrike',
            'nrCalls',
            'meanStrikeCall',
            'sumVolumeCall',
            'sumOpenInterestCall',
            'sumVolumeTimesStrikeCall',
            'weightedStrikeCall',
            'nrPuts',
            'meanStrikePut',
            'sumVolumePut',
            'sumOpenInterestPut',
            'sumVolumeTimesStrikePut',
            'weightedStrikePut',
            'volumeCumSum',
            'openInterestCumSum',
            'nrHigherOptions',
            'higherStrikePriceCum',
            'meanStrikeCallPerc',
            'meanStrikePutPerc',
            'midpointPerc',
            'meanHigherStrike']

features_ext = ['midpointPerc', 'priceDiff', 'priceDiffPerc', 'bidPrice', 'strikePriceCum', 'midpoint', 'lastPrice', 'weightedStrikeCall', 'strikePrice',
            'meanStrikeCall', 'sumVolumeTimesStrikeCall', 'askPrice', 'meanStrikePut', 'daysToExpiration', 'meanStrikeCallPerc',
            'sumVolumeTimesStrikePut', 'weightedStrikePut', 'volumeTimesStrike', 'OBV', 'BBU_5_2.0', 'BBB_5_2.0', 'volatility', 'BBM_5_2.0',
            'sumOpenInterestCall', 'BBL_5_2.0', 'MACD_2_20_9', 'sumOpenInterestPut', 'volumeCumSum', 'MACDs_2_20_9', 'higherStrikePriceCum']

print('Nr of features included: {}'.format(len(features)))

########################
# Split in train and test
# test to split keeping exportedAt column always in same group
X_train, y_train, X_test, y_test, X_val, y_val, X_oot, y_oot = splitDataTrainTestValOot(df, target = 'reachedStrikePrice', date_col='exportedAt', oot_share=0.0, test_share=0.75, val_share=0.75)

#####################
# Train
# Classifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

train_type = 'PROD'
version = 'v1x1'
algorithm = 'CB'
if train_type == 'DEV':
    X_fit = X_train
    y_fit = y_train
    df_test = df_all.loc[X_test.index,:]
    df_test.to_csv("validation/test_df.csv")
elif train_type == 'PROD':
    X_fit = pd.concat([X_train, X_test], sort=False)
    y_fit = pd.concat([y_train, y_test], sort=False)

print('Train data shape: {}'.format(X_fit.shape))
print('Calibration data shape: {}'.format(X_val.shape))
print('Train type: {}\nVersion: {}\nAlgorithm: {}'.format(train_type, version, algorithm))
print('Training uncalibrated model...')

getwd = os.getcwd()
if algorithm == 'AB':
    params = {'n_estimators':1000, 'learning_rate':0.5, 'random_state':42}
    uncal_model = fit_AdaBoost(X_fit[features], y_fit, X_val, y_val, params, save_model = False, ab_path=getwd+'/trained_models/', name='{}_{}{}_{}'.format(train_type, algorithm, n_bits, version))
elif algorithm == 'GB':
    params = {'n_estimators': 3000, 'max_depth': 8, 'max_features': 3, 'min_samples_split': 415, 'subsample': 0.7016649229706161, 'learning_rate': 0.0001877112009793005}
    # sample_weights = getSampleWeights(X_fit, column='exportedAt', normalize=True, squared=False)
    # kwargs = {'sample_weight': sample_weights.values}
    kwargs = {'nothing': 'empty'}
    uncal_model = fit_GBclf(X_fit[features], y_fit, X_val, y_val, params, save_model = False, gbc_path=getwd+'/trained_models/', name='{}_{}{}_{}'.format(train_type, algorithm, n_bits, version), **kwargs)
elif algorithm == 'CB':
    params = {}
    # sample_weights = getSampleWeights(X_fit, column='exportedAt', normalize=True, squared=False)
    # kwargs = {'sample_weight': sample_weights.values}
    kwargs = {'nothing': 'empty'}
    uncal_model = fit_cb(X_fit[features], y_fit, X_val[features], y_val, params, save_model = False, cb_path=getwd+'/trained_models/', name='{}_{}{}_{}'.format(train_type, algorithm, n_bits, version))

print('Training uncalibrated model... Done!')

xVar, yVar, thresholds = roc_curve(y_val, uncal_model.predict_proba(X_val[features])[:, 1])
roc_auc = auc(xVar, yVar)
print('Model has a ROC on validation set of: {}'.format(roc_auc))

print('Calibrate and save model...')
# calibrate and save classifier
cal_model = calibrate_model(uncal_model, X_val[features], y_val, method='sigmoid', save_model=True, path=getwd+'/trained_models/', name='{}_c_{}{}_{}'.format(train_type, algorithm, n_bits, version))


print('Calibrate and save model... Done!')



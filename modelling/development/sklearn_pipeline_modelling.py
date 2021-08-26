import json

from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score, make_scorer, precision_recall_curve, auc, precision_score
from sklearn.metrics import accuracy_score, log_loss
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from option_trading_nonprod.aws import *
from option_trading_nonprod.models.tree_based import *
from option_trading_nonprod.other.trading_strategies import *
from option_trading_nonprod.process.pre_train import *
from option_trading_nonprod.process.stock_price_enriching import *
from option_trading_nonprod.process.train_modifications import *
from option_trading_nonprod.validation.trained_model_validation import *

## functions
class CustomTransformer(BaseEstimator, TransformerMixin):
	def __init__(self, feature_names):
		# print('>init() called.')
		self.feature_names = feature_names
		print('Attached feature names to object')

	def fit(self, X, y = None):
		# print('>fit() called.')
		return self

	def transform(self, X, y = None):
		# print('>transform() called.')
		X_ = X.copy() # creating a copy to avoid changes to original dataset
		# X_ = enrich_df(X_)
		X_.fillna(X_.mean(), inplace=True)
		X_ = X_[self.feature_names]
		print('Features included: {}'.format(len(self.feature_names)))
		return X_

#######################
# Pre set variables and checks
# 32 or 64 bit system
n_bits = 32 << bool(sys.maxsize >> 32)

if platform.system() == 'Darwin':
	s3_profile = 'mrOption'
else:
	s3_profile = 'default'

###### import data from S3
# Set source and target for bucket and keys
today = '2021-04-15'
source_bucket = 'project-option-trading-output'
source_key = 'enriched_data/barchart'.format(today)
# source_key = 'train_data/barchart/enriched_on_{}.csv'.format(today)

# print status of variables
print('Source bucket: {}'.format(source_bucket))
print('Source key: {}'.format(source_key))

# import data
df = load_from_s3(profile=s3_profile, bucket=source_bucket, key_prefix=source_key)
print("Raw imported data shape: {}".format(df.shape))

# Set target
df['reachedStrikePrice'] = np.where(df['maxPrice'] >= df['strikePrice'], 1, 0)
df['percStrikeReached'] = (df['maxPrice'] - df['baseLastPrice']) / (
		df['strikePrice'] - df['baseLastPrice'])
df['finalPriceHigher'] = np.where(df['finalPrice'] >= df['baseLastPrice'], 1, 0)
df['targetV2'] = np.where((df['reachedStrikePrice'] == 1) | (df['finalPriceHigher'] == 1), 1, 0)

df = df.drop_duplicates(subset=['baseSymbol','symbolType','strikePrice','expirationDate','exportedAt'])

df = batch_enrich_df(df, groupByColumns=['exportedAt', 'baseSymbol', 'symbolType', 'expirationDate', 'inTheMoney'])

if 'sector' in df.columns:
	df = df[~df['sector'].isnull()]
	df = pd.get_dummies(df, prefix='sector', columns=['sector'])

# filter set on applicable rows
# only select Call option out of the money
with open('other_files/config_file.json') as json_file:
	config = json.load(json_file)

included_options = config['included_options']

# Filter on basics (like days to expiration and contract type)
df_calls = dfFilterOnGivenSetOptions(df, included_options)
df_calls = df_calls.sort_values('exportedAt', ascending=True)

X_train, y_train, X_test, y_test, X_val, y_val, X_oot, y_oot = splitDataTrainTestValOot(df_calls, target = 'reachedStrikePrice', date_col='exportedAt', oot_share=0.1, test_share=0.8, val_share=0.8)

# Start of tuning
# general approach
# Use all features and create descent model (tiny bit of hyper parameter optimization
# Use trained model to do feature selection
# Use these features to perform hyper paramter optimization with

# features to train on
features_base = ['strikePrice'
	, 'daysToExpiration'
	, 'bidPrice'
	, 'midpoint'
	, 'askPrice'
	, 'lastPrice'
	, 'openInterest'
	, 'volumeOpenInterestRatio'
	, 'volatility'
	, 'volume'
	# to capture end of day price (closer to the price we can buy the stock for)
	, 'firstPrice'
	]

features_info = ['strikePrice'
	, 'baseLastPrice'
	, 'daysToExpiration'
	, 'bidPrice'
	, 'midpoint'
	, 'askPrice'
	, 'lastPrice'
	, 'openInterest'
	, 'volumeOpenInterestRatio'
	, 'volatility'
	, 'volume'
	# , 'marketCap'
	# , 'beta'
	# , 'forwardPE'
	# , 'sector_Healthcare'
	# , 'sector_Technology'
	# , 'sector_Basic Materials'
	# , 'sector_Communication Services'
	# , 'sector_Consumer Cyclical'
	# , 'sector_Consumer Defensive'
	# , 'sector_Energy'
	# , 'sector_Financial Services'
	# , 'sector_Industrials'
	# , 'sector_Real Estate'
	# , 'sector_Services'
	# , 'sector_Utilities'
				 # Features from in batch enriching
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
	, 'meanHigherStrike'
]


features_all = ['strikePrice'
	, 'daysToExpiration'
	, 'bidPrice'
	, 'midpoint'
	, 'askPrice'
	, 'lastPrice'
	, 'openInterest'
	, 'volumeOpenInterestRatio'
	, 'volatility'
	, 'volume'
	# simple calculated features
	, 'priceDiff'
	, 'priceDiffPerc'
	, 'inTheMoney'
	# Features from technical indicators
	, 'MACD_2_20_9'
	, 'MACDh_2_20_9'
	, 'MACDs_2_20_9'
	, 'RSI_14'
	, 'OBV'
	, 'BBL_5_2.0'
	, 'BBM_5_2.0'
	, 'BBU_5_2.0'
	, 'BBB_5_2.0'
	# Features from in batch enriching
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



features_adj = ['midpointPerc', 'priceDiff', 'priceDiffPerc', 'bidPrice',
				'strikePriceCum', 'midpoint', 'lastPrice', 'weightedStrikeCall',
				'strikePrice', 'meanStrikeCall', 'sumVolumeTimesStrikeCall',
				'askPrice', 'meanStrikePut', 'daysToExpiration',
				'meanStrikeCallPerc', 'sumVolumeTimesStrikePut',
				'weightedStrikePut', 'volumeTimesStrike', 'OBV', 'BBU_5_2.0',
				'BBB_5_2.0', 'volatility', 'BBM_5_2.0', 'sumOpenInterestCall',
				'BBL_5_2.0', 'MACD_2_20_9', 'sumOpenInterestPut', 'volumeCumSum',
				'MACDs_2_20_9', 'higherStrikePriceCum']

# topFeatures = feat_imp[feat_imp['importance'] > 0.009]['feature']
features = features_base
# top30features = feat_imp['feature'].head(30)

###########
# Test different classifiers
## creating pipeline
classifiers = [
	AdaBoostClassifier(learning_rate=0.01, n_estimators=2000, random_state=42),
	GradientBoostingClassifier(learning_rate=0.01, n_estimators=1000, min_samples_split=500, max_features='sqrt', max_depth=4, random_state=42, subsample=0.8)
]

# best AUC ROC
# undefined

for classifier in classifiers:
	print(classifier)
	pipe = Pipeline(steps=[('preprocessor', CustomTransformer(features_info)),
					  ('classifier', classifier)])

	# add sample weights
	# sample_weights = getSampleWeights(X_train, column='exportedAt', normalize=True, squared=False)
	# kwargs = {pipe.steps[-1][0] + '__sample_weight': sample_weights}
	# pipe.fit(X_train, y_train, **kwargs)

	# not adding sample weights
	pipe.fit(X_train, y_train)

	print('Validation dataset')
	probs = pipe.predict_proba(X_val)[:,1]
	print("model score: %.3f" % pipe.score(X_val, y_val))
	print("AUC ROC: {}".format(roc_auc_score(y_val, probs)))
	# precision recall auc
	yVar, xVar, thresholds = precision_recall_curve(y_val, probs)
	print("Precision Recall: {}".format(auc(xVar, yVar)))

	print('Test dataset')
	probs = pipe.predict_proba(X_test)[:,1]
	print("model score: %.3f" % pipe.score(X_test, y_test))
	print("AUC ROC: {}".format(roc_auc_score(y_test, probs)))
	# precision recall auc
	yVar, xVar, thresholds = precision_recall_curve(y_test, probs)
	print("Precision Recall: {}".format(auc(xVar, yVar)))

	print('Out of time dataset')
	probs = pipe.predict_proba(X_oot)[:,1]
	print("model score: %.3f" % pipe.score(X_oot, y_oot))
	print("AUC ROC: {}".format(roc_auc_score(y_oot, probs)))
	# precision recall auc
	yVar, xVar, thresholds = precision_recall_curve(y_oot, probs)
	print("Precision Recall: {}".format(auc(xVar, yVar)))


############
# Test different parameters
# parameter names are the <classifier name>__<parameter name> (classifier as named in the pipeline)
gb_param_dist = {
	'classifier__n_estimators': [200],
	'classifier__max_depth': [2,3,4,5],
	'classifier__max_features': [2,4,6],
	'classifier__min_samples_split': [800,1200,1600,2000],
	'classifier__subsample': [0.3,0.5,0.7],
	'classifier__learning_rate': [0.05],
	'classifier__random_state': [42]
}

pipe_cv = Pipeline(steps=[('preprocessor', CustomTransformer(features_adj)),
					   ('classifier', GradientBoostingClassifier())])

grid = GridSearchCV(pipe_cv, param_grid=gb_param_dist, scoring=precision_score, cv=2)
grid.fit(X_train, y_train)

print('Validation dataset')
probs = grid.predict_proba(X_val)[:,1]
print("score = %3.2f" % (grid.score(X_val, y_val)))
print("AUC ROC: {}".format(roc_auc_score(y_val, probs)))
# precision recall auc
yVar, xVar, thresholds = precision_recall_curve(y_val, probs)
print("Precision Recall: {}".format(auc(xVar, yVar)))

print('Test dataset')
probs = grid.predict_proba(X_test)[:,1]
print("score = %3.2f" % (grid.score(X_test, y_test)))
print("AUC ROC: {}".format(roc_auc_score(y_test, probs)))
# precision recall auc
yVar, xVar, thresholds = precision_recall_curve(y_test, probs)
print("Precision Recall: {}".format(auc(xVar, yVar)))

print('Out of time dataset')
probs = grid.predict_proba(X_oot)[:,1]
print("score = %3.2f" % (grid.score(X_oot, y_oot)))
print("AUC ROC: {}".format(roc_auc_score(y_oot, probs)))
# precision recall auc
yVar, xVar, thresholds = precision_recall_curve(y_oot, probs)
print("Precision Recall: {}".format(auc(xVar, yVar)))

print(grid.best_params_)

# best AUC ROC
# model GradientBoostingClassifier({'classifier__learning_rate': 0.01, 'classifier__max_depth': 2, 'classifier__min_samples_split': 16, 'classifier__n_estimators': 1000, 'classifier__random_state': 42, 'classifier__subsample': 0.6})
# features top 20 after fitting all (with within batch enriched)
# midpointPerc, meanStrikeCallPerc, volatility, nrCalls, meanHigherStrike, nrHigherOptions, daysToExpiration, baseLastPrice, bidPrice, volumeCall, higherStrikePriceCum, strikePriceCum, nrPuts, strikePrice, meanStrikeCall, meanStrikePutPerc, volumeCumSum, openInterestPut, openInterestCall, meanStrikePut
# val set:
# Test set:
# Oot set:

best_model = grid.best_estimator.steps[1][1]

#plot feature importance
from option_trading_nonprod.validation.feature_importances import *
featureImportance1(model=best_model, features=features)
feat_imp = featureImportance1(model=pipe.steps[1][1], features=features_info)

# calculate trading profit
simpleTradingStrategy(df_val)

#########
# Calibrate and save model
from option_trading_nonprod.models.calibrate import *
import os
getwd = os.getcwd()
# Calibrate pre trained model
model = pipe.steps[1][1]
model.feature_names = features
train_type = 'DEV'
version = 'v1x4'

if train_type == 'DEV':
	X_fit = X_train
	y_fit = y_train
	df_test.to_csv("data/validation/test_df.csv")
	df_oot.to_csv("data/validation/oot_df.csv")
elif train_type == 'PROD':
	X_fit = pd.concat([X_train, X_test])
	y_fit = pd.concat([y_train, y_test])

X_val.fillna(0, inplace=True)

opunta_params = {'n_estimators': 3000, 'max_depth': 8, 'max_features': 3, 'min_samples_split': 415, 'subsample': 0.7016649229706161, 'learning_rate': 0.0001877112009793005}
params = {'n_estimators': 3000, 'max_depth': 8, 'max_features': 3, 'min_samples_split': 415, 'subsample': 0.7016649229706161, 'learning_rate': 0.0001877112009793005}
model = fit_GBclf(X_train[features_info], y_train, X_val[features_info], y_val, opunta_params, save_model = False, gbc_path=getwd+'/trained_models/', name='GB64_'+version)
Cal_model = calibrate_model(model, X_val, y_val, method='sigmoid', save_model=True, path=getwd + '/trained_models/', name=train_type+'_c_GB64_'+version)

# Create performance HTML
model_name = 'DEV_c_GB64_v1x4'
scraped_since = '2021-06-01'
createModelPerformanceReport(model_name, scraped_since)
from sklearn.model_selection import train_test_split, GroupShuffleSplit

from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.metrics import accuracy_score, log_loss
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV

from option_trading_nonprod.process.stock_price_enriching import *
from option_trading_nonprod.models.calibrate import *
from option_trading_nonprod.models.tree_based import *
from option_trading_nonprod.process.train_modifications import *

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
# Load and prepare data
df = pd.read_csv('data/barchart_yf_enr_1x2.csv')

# Set target
df['reachedStrikePrice'] = np.where(df['maxPrice'] >= df['strikePrice'], 1, 0)
df['percStrikeReached'] = (df['maxPrice'] - df['baseLastPrice']) / (
		df['strikePrice'] - df['baseLastPrice'])
df['finalPriceHigher'] = np.where(df['finalPrice'] >= df['baseLastPrice'], 1, 0)
df['target'] = np.where((df['reachedStrikePrice'] == 1) | (df['finalPriceHigher'] == 1), 1, 0)

df = df.drop_duplicates(subset=['baseSymbol','symbolType','strikePrice','expirationDate','exportedAt'])

df = batch_enrich_df(df, groupByColumns=['exportedAt', 'baseSymbol', 'symbolType', 'expirationDate', 'inTheMoney'])

# filter set on applicable rows
# only select Call option out of the money
df_calls = df[(df['symbolType'] == 'Call') & (df['strikePrice'] > df['baseLastPrice'] * 1.05)].copy()
df_calls = df_calls.sort_values('exportedAt', ascending=True)

# target  used
target = 'reachedStrikePrice'

# Split in train, validation, test and out of time
# Take most recent observations for out of time set (apprx last 5000 observations)

exportDateLast3000 = df_calls.iloc[-3000]['exportedAt']
df_oot = df_calls[df_calls['exportedAt'] >= exportDateLast3000]
df_rest = df_calls.drop(df_oot.index, axis=0).reset_index(drop=True)

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
# Add weights column
X_train = df_train.drop(columns=[target])
y_train = df_train[target]

X_val = df_val.drop(columns=[target])
y_val = df_val[target]

X_test = df_test.drop(columns=[target])
y_test = df_test[target]

X_oot = df_oot.drop(columns=[target])
y_oot = df_oot[target]

print("Train shape: {}\nValidation shape: {}\nTest shape: {}\nOut of time shape: {}".format(X_train.shape,X_val.shape,X_test.shape,X_oot.shape))
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
	pipe = Pipeline(steps=[('preprocessor', CustomTransformer(features)),
					  ('classifier', classifier)])

	# add sample weights
	sample_weights = getSampleWeights(X_train, column='exportedAt', normalize=True, squared=False)
	kwargs = {pipe.steps[-1][0] + '__sample_weight': sample_weights}
	pipe.fit(X_train, y_train, **kwargs)

	print('Validation dataset')
	probs = pipe.predict_proba(X_val)[:,1]
	print("model score: %.3f" % pipe.score(X_val, y_val))
	print("AUC ROC: {}".format(roc_auc_score(y_val, probs)))

	print('Test dataset')
	probs = pipe.predict_proba(X_test)[:,1]
	print("model score: %.3f" % pipe.score(X_test, y_test))
	print("AUC ROC: {}".format(roc_auc_score(y_test, probs)))

	print('Out of time dataset')
	probs = pipe.predict_proba(X_oot)[:,1]
	print("model score: %.3f" % pipe.score(X_oot, y_oot))
	print("AUC ROC: {}".format(roc_auc_score(y_oot, probs)))


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

grid = GridSearchCV(pipe_cv, param_grid=gb_param_dist, scoring=make_scorer(roc_auc_score), cv=2)
grid.fit(X_train, y_train)

print('Validation dataset')
probs = grid.predict_proba(X_val)[:,1]
print("score = %3.2f" % (grid.score(X_val, y_val)))
print("AUC ROC: {}".format(roc_auc_score(y_val, probs)))

print('Test dataset')
probs = grid.predict_proba(X_test)[:,1]
print("score = %3.2f" % (grid.score(X_test, y_test)))
print("AUC ROC: {}".format(roc_auc_score(y_test, probs)))

print('Out of time dataset')
probs = grid.predict_proba(X_oot)[:,1]
print("score = %3.2f" % (grid.score(X_oot, y_oot)))
print("AUC ROC: {}".format(roc_auc_score(y_oot, probs)))
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
feat_imp = featureImportance1(model=pipe.steps[1][1], features=features_all)

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
version = 'v3x3'

if train_type == 'DEV':
	X_fit = X_train
	y_fit = y_train
	df_test.to_csv("data/validation/test_df.csv")
	df_oot.to_csv("data/validation/oot_df.csv")
elif train_type == 'PROD':
	X_fit = pd.concat([X_train, X_test])
	y_fit = pd.concat([y_train, y_test])

X_val.fillna(0, inplace=True)

opunta_params = {'n_estimators': 3000, 'max_depth': 10, 'max_features': 8, 'min_samples_split': 225, 'subsample': 0.808392563444737, 'learning_rate': 0.00010030663168798627}
params = {'n_estimators': 3000, 'max_depth': 4, 'max_features': 12, 'min_samples_split': 300, 'subsample': 0.853500248686749, 'learning_rate': 0.005}
model = fit_GBclf(X_train[features_adj], y_train, X_val[features_adj], y_val, opunta_params, save_model = False, gbc_path=getwd+'/trained_models/', name='GB64_'+version)
Cal_model = calibrate_model(model, X_val, y_val, method='sigmoid', save_model=True, path=getwd + '/trained_models/', name=train_type+'_c_GB64_'+version)

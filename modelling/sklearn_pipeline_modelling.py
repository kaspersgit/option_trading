from sklearn.model_selection import train_test_split, GroupShuffleSplit

from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.metrics import accuracy_score, log_loss
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV

from option_trading_nonprod.process.stock_price_enriching import *

## functions
class CustomTransformer(BaseEstimator, TransformerMixin):
  # add another additional parameter, just for fun, while we are at it
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
		# X_.fillna(0, inplace=True)
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

df = df.drop_duplicates(subset=['baseSymbol','symbolType','strikePrice','expirationDate','exportedAt'])

df = batch_enrich_df(df, groupByColumns=['exportedAt', 'baseSymbol', 'symbolType', 'expirationDate', 'inTheMoney'])

# filter set on applicable rows
# only select Call option out of the money
df_calls = df[(df['symbolType'] == 'Call') & (df['strikePrice'] > df['baseLastPrice'])].copy()
df_calls = df_calls.sort_values('exportedAt', ascending=True)

# target  used
target = 'reachedStrikePrice'

# Split in train, validation, test and out of time
# Take most recent observations for out of time set (apprx last 5000 observations)
exportDateLast5000 = df_calls.iloc[-5000]['exportedAt']
df_oot = df_calls[df_calls['exportedAt'] >= exportDateLast5000]
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
X_train = df_train.drop(columns=[target])
y_train = df_train[target]

X_val = df_val.drop(columns=[target])
y_val = df_val[target]

X_test = df_test.drop(columns=[target])
y_test = df_test[target]

X_oot = df_oot.drop(columns=[target])
y_oot = df_oot[target]

# features to train on
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
	# variables under are batch related
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

###########
# Test different classifiers
## creating pipeline
classifiers = [
	# KNeighborsClassifier(3),
	# SVC(kernel="rbf", C=0.025, probability=True),
	# NuSVC(probability=True),
	# MLPClassifier(),
	# RandomForestClassifier(n_estimators=1000, min_samples_leaf = 50, random_state=42),
	AdaBoostClassifier(learning_rate=0.01, n_estimators=2000, random_state=42),
	# GradientBoostingClassifier(learning_rate=0.005, n_estimators=2000, min_samples_split=8, max_depth=4 , random_state=42, subsample=0.8),
	GradientBoostingClassifier(learning_rate=0.01, n_estimators=1000, min_samples_split=8, max_depth=4, random_state=42, subsample=1)
]

# best AUC ROC
# model GradientBoostingClassifier(learning_rate=0.01, n_estimators=3000, min_samples_split=4, max_depth=4 , random_state=42, subsample=0.8)
# all features
# Test set: 0.816
# Oot set: 0.852

for classifier in classifiers:
	print(classifier)
	pipe = Pipeline(steps=[('preprocessor', CustomTransformer(features_all)),
					  ('classifier', classifier)])
	pipe.fit(X_train, y_train)
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
rf_param_dist = {
	'classifier__n_estimators': [500,1000],
	'classifier__learning_rate': [0.01,0.05,0.1]
}

ab_param_dist = {
	'classifier__n_estimators': [500,1000],
	'classifier__learning_rate': [0.01,0.05,0.1]
}

gb_param_dist = {
	'classifier__n_estimators': [500,1000],
	'classifier__learning_rate': [0.01],
	'classifier__min_samples_split': [4,10]
}
gb_param_dist = {
	'classifier__n_estimators': [3000],
	'classifier__max_depth': [4,6],
	'classifier__min_samples_split': [4,8],
	'classifier__subsample': [0.8,1],
	'classifier__learning_rate': [0.01, 0.005],
	'classifier__random_state': [42]
}

pipe_cv = Pipeline(steps=[('preprocessor', CustomTransformer(features)),
					   ('classifier', GradientBoostingClassifier())])

grid = GridSearchCV(pipe_cv, param_grid=gb_param_dist, scoring=make_scorer(roc_auc_score), cv=2)
grid.fit(X_train, y_train)

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
# model GradientBoostingClassifier('classifier__learning_rate': 0.01, 'classifier__max_depth': 6, 'classifier__min_samples_split': 4, 'classifier__n_estimators': 3000, 'classifier__subsample': 0.8, 'classifier__random_state': 42)
# features top 20 after fitting all (with within batch enriched)
# midpointPerc, meanStrikeCallPerc, volatility, nrCalls, meanHigherStrike, nrHigherOptions, daysToExpiration, baseLastPrice, bidPrice, volumeCall, higherStrikePriceCum, strikePriceCum, nrPuts, strikePrice, meanStrikeCall, meanStrikePutPerc, volumeCumSum, openInterestPut, openInterestCall, meanStrikePut
# Test set: 0.816
# Oot set: 0.846

best_model = best_estimator.steps[1][1]

#plot feature importance
from option_trading_nonprod.validation.feature_importances import *
featureImportance1(model=best_model, features=features)
feat_imp = featureImportance1(model=pipe.steps[1][1], features=features_all)
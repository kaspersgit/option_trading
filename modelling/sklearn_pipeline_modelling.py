from sklearn.model_selection import train_test_split

from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

from option_trading_nonprod.process.stock_price_enriching import *

## functions
class CustomTransformer(BaseEstimator, TransformerMixin):
  # add another additional parameter, just for fun, while we are at it
	def __init__(self, feature_names):
		print('\n>>>>>>>init() called.\n')
		self.feature_names = feature_names

	def fit(self, X, y = None):
		print('\n>>>>>>>fit() called.\n')
		return self

	def transform(self, X, y = None):
		print('\n>>>>>>>transform() called.\n')
		X_ = X.copy() # creating a copy to avoid changes to original dataset
		X_ = enrich_df(X_)
		X_ = X_[self.feature_names]
		return X_

# clean out duplicates to be sure
df_all = df_all.drop(axis=1, columns='Unnamed: 0')
df_all = df_all.drop_duplicates(subset=['baseSymbol','symbolType','strikePrice','expirationDate','exportedAt'])

# Add internal (within same batch) information
df_all = enrich_df(df_all)

# Set target
df_all['reachedStrikePrice'] = np.where(df_all['maxPrice'] >= df_all['strikePrice'],1,0)
df_all['percStrikeReached'] = (df_all['maxPrice'] - df_all['baseLastPrice']) / (df_all['strikePrice'] - df_all['baseLastPrice'])

# filter set on applicable rows
# only select Call option out of the money
df = df_all[(df_all['symbolType'] == 'Call') & (df_all['strikePrice'] > df_all['baseLastPrice'])]


## creating pipeline
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=0.025, probability=True),
    NuSVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier()
    ]

#######################
# Load and prepare data
df = pd.read_csv('data/barchart_yf_enr_1.csv')

# Set target
df['reachedStrikePrice'] = np.where(df['maxPrice'] >= df['strikePrice'], 1, 0)
df['percStrikeReached'] = (df['maxPrice'] - df['baseLastPrice']) / (
		df['strikePrice'] - df['baseLastPrice'])

df = df.drop_duplicates(subset=['baseSymbol','symbolType','strikePrice','expirationDate','exportedAt'])
# filter set on applicable rows
# only select Call option out of the money
df_calls = df[(df['symbolType'] == 'Call') & (df['strikePrice'] > df['baseLastPrice'])].copy()

# target  used
target = 'reachedStrikePrice'

# Split in train, validation, test and out of time
df_oot = df_calls.sort_values('exportedAt', ascending=True)[-5000::]
df_rest = df_calls.drop(df_oot.index, axis=0)

df_train, df_test = train_test_split(df_rest, test_size=0.25, random_state=42)
df_train, df_val = train_test_split(df_train, test_size=0.25, random_state=42)

# clean unwanted columns for model training
X_train = df_train.drop(columns=[target])
y_train = df_train[target]

X_val = df_val.drop(columns=[target])
y_val = df_val[target]

X_test = df_test.drop(columns=[target])
y_test = df_test[target]

# features to train on
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
    , 'volatility']

############
# Test different parameters
# parameter names are the <classifier name>__<parameter name> (classifier as named in the pipeline)
param_dist = {
 'classifier__n_estimators': [500,1000],
 'classifier__learning_rate': [0.01,0.05,0.1]
 }

pipe = Pipeline(steps=[('preprocessor', CustomTransformer(features)),
					   ('classifier', AdaBoostClassifier())])

grid = GridSearchCV(pipe, param_grid=param_dist, scoring=make_scorer(roc_auc_score), cv=5)
grid.fit(X_train, y_train)

print("score = %3.2f" %(grid.score(X_test,y_test)))
print(grid.best_params_)

###########
# Test different classifiers
for classifier in classifiers:
    pipe = Pipeline(steps=[('preprocessor', CustomTransformer(features)),
                      ('classifier', classifier)])
    pipe.fit(X_train, y_train)
    print(classifier)
    print("model score: %.3f" % pipe.score(X_test, y_test))
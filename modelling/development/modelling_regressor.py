import matplotlib.pyplot as plt
import numpy as np
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
import os
from sklearn.model_selection import train_test_split

from option_trading_nonprod.models.calibrate import *
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
df_all['percStrikeReached'] = (df_all['maxPrice'] - df_all['baseLastPrice']) / (df_all['strikePrice'] - df_all['baseLastPrice'])

# filter set on applicable rows
# only select Call option out of the money
df = df_all[(df_all['symbolType'] == 'Call') & (df_all['strikePrice'] > df_all['baseLastPrice'])]

# feature selection
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

########################
# Split in train, validation, test and out of time
df_oot = df.sort_values('exportedAt', ascending=True)[-5000::]
df_rest = df.drop(df_oot.index, axis=0)

df_train, df_test = train_test_split(df_rest, test_size=0.25, random_state=42)
df_train, df_val = train_test_split(df_train, test_size=0.25, random_state=42)

# clean unwanted columns for model training
X_train = df_train[features]
y_train = df_train['percStrikeReached']

X_val = df_val[features]
y_val = df_val['percStrikeReached']

X_test = df_test[features]
y_test = df_test['percStrikeReached']

#####################
# Train and predict
# AdaBoost classifier
# either DEV(ELOPMENT) or PROD(UCTION)
# v1x0 trained on data with max expirationDate 2020-10-30
# v1x1 trained on data with max expirationDate 2020-12-18

train_type = 'DEV'
version = 'v1x3'
if train_type == 'DEV':
    X_fit = X_train
    y_fit = y_train
    df_test = df_all.loc[X_test.index,:]
    df_test.to_csv("data/validation/test_df.csv")
    df_oot.to_csv("data/validation/oot_df.csv")
elif train_type == 'PROD':
    X_fit = pd.concat([X_train, X_test])
    y_fit = pd.concat([y_train, y_test])

params = {'n_estimators': 500,
          'max_depth': 4,
          'min_samples_split': 5,
          'learning_rate': 0.01,
          'loss': 'ls'}

reg = ensemble.GradientBoostingRegressor(**params)
reg.fit(X_fit, y_fit)

r_pred = reg.predict(X_test)

df_test['r_pred'] = r_pred

mse = mean_squared_error(y_test, r_pred)
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))


# plot prediction vs actual
fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.scatter(df_test['r_pred'], df_test['percStrikeReached'], s = 7, color='g', alpha=0.7, label='Increase wrt strike price')
ax.legend(loc="upper right")
ax.set_xlim(-3,8)
ax.set_ylim(-1,8)
ax.set_xlabel('Perc of Strike predicted')
ax.set_ylabel('Actual')
ax.set_title('All Call options plotted')
plt.show()

########
# estimating profitability
df_test['stocksBought'] = 100 / df_test['baseLastPrice']
df_test['cost'] = df_test['stocksBought'] * df_test['baseLastPrice']
df_test['revenue'] = df_test['stocksBought'] * np.where(df_test['percStrikeReached'] >= df_test['r_pred'], df_test['r_pred'], df_test['finalPrice'])
df_test['profit'] = df_test['revenue'] - df_test['cost']
df_test['profitPerc'] = df_test['profit'] / df_test['cost']
df_test['reachedPrediction'] = np.where(df_test['percStrikeReached'] >= df_test['r_pred'], 1, 0)

df_test['profit'].sum()
# below gives big profit?
minThreshold = 1.5
maxThreshold = 9
total_profit = df_test[(df_test['r_pred'] > minThreshold) & (df_test['r_pred'] < maxThreshold)]['profit'].sum()
total_cost = df_test[(df_test['r_pred'] > minThreshold) & (df_test['r_pred'] < maxThreshold)]['cost'].sum()
print(total_profit / total_cost)
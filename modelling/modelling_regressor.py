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
features = ['exportedAt'
    , 'percStrikeReached'
    , 'baseLastPrice'
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



df = df[features]

########################
# Split in train, validation, test and out of time
df_oot = df.sort_values('exportedAt', ascending=True)[-5000::]
df_rest = df.drop(df_oot.index, axis=0)

# clean unwanted columns for model training
df_oot = df_oot.drop(columns=['baseSymbol','symbolType','tradeTime','exportedAt','expirationDate', 'minPrice', 'maxPrice',
       'finalPrice', 'firstPrice'], errors='ignore')
df_rest = df_rest.drop(columns=['baseSymbol','symbolType','tradeTime','exportedAt','expirationDate', 'minPrice', 'maxPrice',
       'finalPrice', 'firstPrice'], errors='ignore')

X = df_rest.drop(columns='percStrikeReached')
y = df_rest['percStrikeReached']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

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

mse = mean_squared_error(y_test, r_pred)
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))

X_test['r_pred'] = r_pred
X_test['r_actual'] = y_test

# plot prediction vs actual
fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.scatter(X_test['r_pred'], X_test['r_actual'], s = 7, color='g', alpha=0.7, label='Increase wrt strike price')
ax.legend(loc="upper right")
ax.set_xlim(-3,8)
ax.set_ylim(-1,8)
ax.set_xlabel('Perc of Strike predicted')
ax.set_ylabel('Actual')
ax.set_title('All Call options plotted')
plt.show()

########
# estimating profitability
df = X_test
df['stocksBought'] = 100 / df['baseLastPrice']
df['cost'] = df['stocksBought'] * df['baseLastPrice']
df['revenue'] = df['stocksBought'] * np.where(df['r_actual'] >= df['r_pred'], df['r_pred'], df['baseLastPrice'] * 0.9)
df['profit'] = df['revenue'] - df['cost']
df['profitPerc'] = df['profit'] / df['cost']
df['reachedPrediction'] = np.where(df['r_actual'] >= df['r_pred'], 1, 0)

df['profit'].sum()
# below gives big profit?
minThreshold = 2
maxThreshold = 4
total_profit = df[(df['r_pred'] > minThreshold) & (df['r_pred'] < maxThreshold)]['profit'].sum()
total_cost = df[(df['r_pred'] > minThreshold) & (df['r_pred'] < maxThreshold)]['cost'].sum()
print(total_profit / total_cost)
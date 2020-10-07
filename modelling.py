#%%
import yfinance as yf
import pandas as pd
from datetime import datetime
import numpy as np
import pickle

from option_trading_nonprod.process.merge_and_clean import *
from option_trading_nonprod.process.stock_price_enriching import *
from option_trading_nonprod.models import *
# %%


def enrich_df(df):
    df['priceDiff'] = df['strikePrice'] - df['baseLastPrice']
    df['priceDiffPerc'] = df['strikePrice'] / df['baseLastPrice']
    df['inTheMoney'] = np.where((df['symbolType']=='Call') & (df['baseLastPrice'] >= df['strikePrice']),1,0)
    df['inTheMoney'] = np.where((df['symbolType']=='Putt') & (df['baseLastPrice'] <= df['strikePrice']),1,df['inTheMoney'])
    df['nrOptions'] = 1
    df['strikePriceCum'] = df['strikePrice']

    df.sort_values(['exportedAt','baseSymbol','symbolType','expirationDate','strikePrice'
        ], inplace=True)

    df_symbol = df[['exportedAt','baseSymbol','symbolType','expirationDate','strikePrice','inTheMoney','volume','openInterest'
            ]].groupby(['exportedAt','baseSymbol','symbolType','expirationDate','inTheMoney'
            ]).agg({'baseSymbol':'count', 'strikePrice':'mean', 'volume':'sum', 'openInterest':'sum'
            }).rename(columns={'baseSymbol':'nrOccurences', 'strikePrice':'meanStrikePrice'
            }).reset_index()
    
    # only give info about calls with higher strike price
    df_option_inv_cum = df[['exportedAt','baseSymbol','symbolType','expirationDate','strikePrice','strikePriceCum','inTheMoney','volume','openInterest','nrOptions'
        ]].groupby(['exportedAt','baseSymbol','symbolType','expirationDate','inTheMoney','strikePrice'
        ]).sum().sort_values('strikePrice', ascending = False
        ).groupby(['exportedAt','baseSymbol','symbolType','expirationDate','inTheMoney']
        ).agg({'volume':'cumsum', 'openInterest':'cumsum', 'nrOptions':'cumsum', 'strikePriceCum':'cumsum'
        }).rename(columns={'volume':'volumeCumSum', 'openInterest':'openInterestCumSum', 'nrOptions':'nrHigherOptions', 'strikePriceCum':'higherStrikePriceCum'
        }).reset_index()
    
    df_call = df_symbol[df_symbol['symbolType']=='Call']
    df_call.rename(columns={'nrOccurences':'nrCalls', 'meanStrikePrice':'meanStrikeCall', 'volume':'volumeCall', 'openInterest':'openInterestCall'}, inplace=True)
    df_call.drop(columns=['symbolType'], inplace=True)
    df_put = df_symbol[df_symbol['symbolType']=='Put']
    df_put.rename(columns={'nrOccurences':'nrPuts', 'meanStrikePrice':'meanStrikePut', 'volume':'volumePut', 'openInterest':'openInterestPut'}, inplace=True)
    df_put.drop(columns=['symbolType'], inplace=True)

    # Add summarized data from Calls and Puts to df
    df = pd.merge(df,df_call, how='left', on=['exportedAt','baseSymbol','expirationDate','inTheMoney'])
    df = pd.merge(df,df_put, how='left', on=['exportedAt','baseSymbol','expirationDate','inTheMoney'])
    df = pd.merge(df,df_option_inv_cum, how='left', on=['exportedAt','baseSymbol','symbolType','expirationDate','inTheMoney','strikePrice'])
    # set nr of occurences to 0 when NaN
    df[['nrCalls','nrPuts','volumeCall','volumePut']] = df[['nrCalls','nrPuts','volumeCall','volumePut']].fillna(0)

    df['meanStrikeCallPerc'] = df['meanStrikeCall'] / df['baseLastPrice']
    df['meanStrikePutPerc'] = df['meanStrikePut'] / df['baseLastPrice']
    df['midpointPerc'] = df['midpoint'] / df['baseLastPrice']
    df['meanHigherStrike'] = df['higherStrikePriceCum'] / df['nrHigherOptions']
    return(df)
# %%

#%%
# Load and clean data
# import market beat
mb20200624 = pd.read_csv('/Users/kasper.de-harder/gits/option_trading/marketbeat_call_activity_2020-06-24.csv')
mb20200625 = pd.read_csv('/Users/kasper.de-harder/gits/option_trading/marketbeat_call_activity_2020-06-25.csv')
mb20200629 = pd.read_csv('/Users/kasper.de-harder/gits/option_trading/marketbeat_call_activity_2020-06-29.csv')
mb20200701 = pd.read_csv('/Users/kasper.de-harder/gits/option_trading/marketbeat_call_activity_2020-07-01.csv')
mb20200706 = pd.read_csv('/Users/kasper.de-harder/gits/option_trading/marketbeat_call_activity_2020-07-06.csv')
mb20200707 = pd.read_csv('/Users/kasper.de-harder/gits/option_trading/marketbeat_call_activity_2020-07-07.csv')
mb20200709 = pd.read_csv('/Users/kasper.de-harder/gits/option_trading/marketbeat_call_activity_2020-07-09.csv')
mb20200710 = pd.read_csv('/Users/kasper.de-harder/gits/option_trading/marketbeat_call_activity_2020-07-10.csv')
mb20200714 = pd.read_csv('/Users/kasper.de-harder/gits/option_trading/marketbeat_call_activity_2020-07-14.csv')
mb20200715 = pd.read_csv('/Users/kasper.de-harder/gits/option_trading/marketbeat_call_activity_2020-07-15.csv')
mb20200716 = pd.read_csv('/Users/kasper.de-harder/gits/option_trading/marketbeat_call_activity_2020-07-16.csv')
mb20200717 = pd.read_csv('/Users/kasper.de-harder/gits/option_trading/marketbeat_call_activity_2020-07-17.csv')


# import barchart
df20200624 = pd.read_csv('/Users/kasper.de-harder/gits/option_trading/barchart_unusual_activity_2020-06-24.csv')
df20200625 = pd.read_csv('/Users/kasper.de-harder/gits/option_trading/barchart_unusual_activity_2020-06-25.csv')
df20200629 = pd.read_csv('/Users/kasper.de-harder/gits/option_trading/barchart_unusual_activity_2020-06-29.csv')
df20200701 = pd.read_csv('/Users/kasper.de-harder/gits/option_trading/barchart_unusual_activity_2020-07-01.csv')
df20200703 = pd.read_csv('/Users/kasper.de-harder/gits/option_trading/barchart_unusual_activity_2020-07-03.csv')
df20200706 = pd.read_csv('/Users/kasper.de-harder/gits/option_trading/barchart_unusual_activity_2020-07-06.csv')
df20200707 = pd.read_csv('/Users/kasper.de-harder/gits/option_trading/barchart_unusual_activity_2020-07-07.csv')
df20200709 = pd.read_csv('/Users/kasper.de-harder/gits/option_trading/barchart_unusual_activity_2020-07-09.csv')
df20200710 = pd.read_csv('/Users/kasper.de-harder/gits/option_trading/barchart_unusual_activity_2020-07-10.csv')
df20200714 = pd.read_csv('/Users/kasper.de-harder/gits/option_trading/barchart_unusual_activity_2020-07-14.csv')
df20200715 = pd.read_csv('/Users/kasper.de-harder/gits/option_trading/barchart_unusual_activity_2020-07-15.csv')
df20200716 = pd.read_csv('/Users/kasper.de-harder/gits/option_trading/barchart_unusual_activity_2020-07-16.csv')
df20200717 = pd.read_csv('/Users/kasper.de-harder/gits/option_trading/barchart_unusual_activity_2020-07-17.csv')
df20200722 = pd.read_csv('/Users/kasper.de-harder/gits/option_trading/barchart_unusual_activity_2020-07-22.csv')
df20200723 = pd.read_csv('/Users/kasper.de-harder/gits/option_trading/barchart_unusual_activity_2020-07-23.csv')
df20200728 = pd.read_csv('/Users/kasper.de-harder/gits/option_trading/barchart_unusual_activity_2020-07-28.csv')

df = pd.concat([df20200624,df20200625,df20200629,df20200701,df20200703,df20200706,df20200707],ignore_index=True)
mb_df = pd.concat([mb20200624,mb20200625,mb20200629,mb20200701,mb20200706,mb20200707],ignore_index=True)

cols = ['volume','openInterest']
df = commas2points2float(df, cols)

# Newer so above is already applied in scraping script
df = pd.concat([df,df20200709,df20200710,df20200714,df20200715,df20200716,df20200717],ignore_index=True)
df['exportedAt'] = pd.to_datetime(df['exportedAt']).dt.strftime('%Y-%m-%d')
cols = ['volatility']
df = commas2points2float(df, cols)

#%%
# Adding some additional columns
df = enrich_df(df)
#%%
# reducing size of dataset- only select interesting options
df = get_mature_df(df)
df = df[df['inTheMoney']==0]
# Adding the stockprices
df_price_enr = add_stock_price(df)
df_pastprice_enr = last_10d_avg(df)

# %%
# Merge into big df
df_enr = pd.merge(df,df_price_enr, how='left', on=['baseSymbol','expirationDate','exportedAt'])
df_enr = pd.merge(df_enr, df_pastprice_enr[['baseSymbol','exportedAt','meanLast10D']], how='left', on=['baseSymbol','exportedAt'])

#df_enr = pd.merge(df_enr, df_pastprice_enr, how='left', left_on=['baseSymbol','exportedAt'], right_on=['ticker','dataDate'])

# Add target variables
# What price we think we can buy it for?
# baseLastPrice ?
# nextBDopen ?
buyPrice = 'baseLastPrice' 
# Add flag if stock raised 50%
df_enr['high_plus50p'] = np.where(df_enr[buyPrice] * 1.5 <= df_enr['maxPrice'],1,0)
# Add if stock reached 110% or 90% of start price
df_enr['high_plus10p'] = np.where(df_enr[buyPrice] * 1.1 <= df_enr['maxPrice'],1,0)
df_enr['low_min10p'] = np.where(df_enr[buyPrice] * 0.9 >= df_enr['minPrice'],1,0)
# Too be safe, if reached both targets, put 110% target to zero
df_enr['high_plus10p'] = np.where(df_enr['low_min10p'] == 1,0,df_enr['high_plus10p'])
# Add flag if they reached strikeprice
df_enr['reachedStrike'] = np.where(df_enr['maxPrice'] >= df_enr['strikePrice'],1,0)
# Add flag if reached 110% of strikeprice
df_enr['reachedStrike110p'] = np.where(df_enr['maxPrice'] >= 1.1*df_enr['strikePrice'],1,0)

#%%%
# Get profit 
df_enr['revenue'] = np.where(df_enr['high_plus10p'] == 1, 1.1*df_enr[buyPrice], df_enr['lastClose'])
df_enr['revenue'] = np.where(df_enr['low_min10p'] == 1, 0.9*df_enr[buyPrice], df_enr['revenue'])
df_enr['profit'] = df_enr['revenue'] - df_enr[buyPrice]
df_enr['profitPerc'] = df_enr['profit']/df_enr[buyPrice]


df_enr['revenueStrike'] = np.where(df_enr['reachedStrike'] == 1, df_enr['strikePrice'], df_enr['lastClose'])
df_enr['revenueStrike'] = np.where(df_enr['low_min10p'] == 1, 0.9*df_enr[buyPrice], df_enr['revenueStrike'])
df_enr['profitStrike'] = df_enr['revenueStrike'] - df_enr[buyPrice]
df_enr['profitStrikePerc'] = df_enr['profitStrike']/df_enr[buyPrice]
df_enr['revenueStrike110p'] = np.where(df_enr['reachedStrike110p'] == 1, 1.1 * df_enr['strikePrice'], df_enr['lastClose'])
df_enr['revenueStrike110p'] = np.where(df_enr['low_min10p'] == 1, 0.9*df_enr[buyPrice], df_enr['revenueStrike110p'])
df_enr['profitStrike110p'] = df_enr['revenueStrike110p'] - df_enr[buyPrice]
df_enr['profitStrikePerc110p'] = df_enr['profitStrike110p']/df_enr[buyPrice]
df_enr['units'] = 1000/df_enr[buyPrice]
df_enr['weightedProfitStrike110p'] = df_enr['units'] * df_enr['profitStrike110p']
#%%
# Select only mature cases (and exclude options with less then 5 days to expiration)
df_mature_call = get_mature_df(df_enr)
df_mature_call.describe()

# TODO aggregate options of the same company 
# now extra weight is on characteristics of e.g. TESLA 
# as that stock rose and it had a lot of options showing activity

#%%
# Predicting
# Only include data points in regression we would act on in real life
df_regr = up_for_trade(df_mature_call)
df_regr = df_regr[df_regr['daysToExpiration'] < 16]

# variable to be predicted
target = 'reachedStrike'
# input used to predict with
ex_vars = [#'baseLastPrice',
       #'strikePrice',
       #'volOIrate',
        'daysToExpiration',
        'lastPrice',
        #'volume',
        'openInterest',
       'volumeOpenInterestRatio',
        'volatility',
       'priceDiff',
        'priceDiffPerc',
        'nrCalls',
        'meanStrikeCall',
        #'volumeCall',
        'openInterestCall',
        'nrPuts',
        'volumeCumSum', 
        'openInterestCumSum',
       'nrHigherOptions', 
       'higherStrikePriceCum',
        'meanStrikeCallPerc',
        'meanHigherStrike',
        'midpointPerc',
       #'nextBDopen'
       ]

#train_set = df_regr.sample(frac=0.85, random_state=1)
train_set = df_regr[df_regr['exportedAt']<'2020-07-14']
test_set = df_regr.drop(train_set.index).reset_index(drop=True)

# Training logistic regression
logit_preds, logit_model = logitModel(test_set, train_set, ex_vars, target=target)
#rf_preds, rf_model = RandomForest(test_set, train_set, ex_vars, target=target)

# mean profitability of cases with prediction > 50%
# Assuming we would invest a certain amount equally spread among the stocks
predicted_df = logit_preds #.copy()
threshold = 0.5
filtered_df = predicted_df[(predicted_df['prediction']>threshold) 
    & (1.05 * predicted_df[buyPrice] < predicted_df['strikePrice'])
    ][['baseSymbol','exportedAt','expirationDate','strikePrice',buyPrice,'maxPrice','profitStrike','profitStrike110p','lastClose','prediction']]
filtered_df['units'] = 10000/filtered_df[buyPrice]
if target == 'reachedStrik110p':
    profitVar = 'profitStrike110p'
elif target == 'reachedStrike':
    profitVar = 'profitStrike'
filtered_df['weightedProfit'] = filtered_df[profitVar] * filtered_df['units']
filtered_mean = filtered_df.mean()
print('Profit margin: {} \nActing on {} call options'.format(round(filtered_mean['weightedProfit'] / 10000,4),len(filtered_df)))

# Showing the ROC and AUC
from sklearn import metrics
auc = metrics.roc_auc_score(predicted_df[target], predicted_df['prediction'])
print("The Area under the ROC is {}".format(round(auc,3)))\
# to visualize
import matplotlib.pyplot as plt
fpr, tpr, thresholds = metrics.roc_curve(predicted_df[target], predicted_df['prediction'])
plt.plot(fpr, tpr)

#%%
threshold = 0.7
# to filter out already into the money options
filtered_set = logit_preds[(logit_preds['inTheMoney']==0) 
    & (logit_preds['priceDiffPerc'] > 1.05) 
    #& (logit_preds['strikePrice'] > logit_preds[buyPrice])
    #& (logit_preds['strikePrice'] < 1000)
 ]
filtered_set[filtered_set['prediction']>threshold].mean()

# %%
filtered_set[filtered_set['prediction']>threshold].describe()

# %%
data = yf.download('CLDR', start='2020-06-23', end='2020-07-01')
data
# %%
# IF happy save models
# LOGIT
logit_model.save('/Users/kasper.de-harder/gits/option_trading/modelLogit')

#%%
# RandomForest
# save the model to disk
filename = '/Users/kasper.de-harder/gits/option_trading/RandomForest.sav'
pickle.dump(rf_model, open(filename, 'wb'))

# %%

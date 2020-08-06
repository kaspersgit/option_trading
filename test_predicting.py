# Predicting which stocks have the best odds of being profitable
#%%
# packages
import pandas as pd
from statsmodels.discrete.discrete_model import LogitResults
import numpy as np
from helper_functions import *
import pickle

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
    df['volOIrate'] = df['openInterestCall'] / df['volumeCall']
    return(df)


# %%
# Load newest data
df = pd.read_csv('/Users/kasper.de-harder/gits/option_trading/barchart_unusual_activity_2020-07-28.csv')

# Adding some additional columns
df = enrich_df(df)
# In case we are predicting the stocks and not options 
df = df.sort_values(['strikePrice','expirationDate'])
df = df.drop_duplicates(subset=['baseSymbol','symbolType','exportedAt'])
df['const'] = 1.0
#%%
# Load model
model = LogitResults.load('/Users/kasper.de-harder/gits/option_trading/modelLogitStock')
#model = pickle.load(open('/Users/kasper.de-harder/gits/option_trading/RandomForest.sav', 'rb'))
# Select columns which are model needs as input but leave out the constant
cols = model.params.index

pred = model.predict(df[cols])
df['prediction'] = pred

# %%
threshold = 0.2
buy_advise = df[(df['prediction'] > threshold) & 
    (df['symbolType']=='Call') & 
    (df['daysToExpiration'] < 40) & 
    (df['priceDiffPerc'] > 1.03) & 
    (df['daysToExpiration'] > 3) & 
    (df['strikePrice'] < 200)]
buy_advise = buy_advise[['baseSymbol', 'expirationDate', 'baseLastPrice', 'strikePrice', 'priceDiffPerc', 'prediction']]
buy_advise = buy_advise.sort_values('prediction').reset_index(drop=True)
# %%
buy_advise

# %%

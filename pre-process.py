# Load in all csv files from source folder
import os
import pandas as pd
from option_trading_nonprod.process import *

# Temp setting
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Set path
cwd = os.getcwd()
directory = os.path.join(cwd,"data/barchart")

# create empty df
df = pd.DataFrame()
# loop through all csv files and concatonate
for root,dirs,files in os.walk(directory):
    for file in files:
       if file.endswith(".csv"):
           f = pd.read_csv(os.path.join(root,file))
           #  Concatenate files into one pandas df
           df = pd.concat([df,f])
df.reset_index(drop=True, inplace=True)

# Select columns
df = df[['baseSymbol', 'baseLastPrice', 'symbolType', 'strikePrice', 'expirationDate', 'daysToExpiration', 'bidPrice', 'midpoint', 'askPrice', 'lastPrice', 'volume', 'openInterest', 'volumeOpenInterestRatio', 'volatility', 'tradeTime', 'exportedAt']]

# filter on only mature options
df = df[df['expirationDate'] < today]

# Helper functions
def colType2Float(series, decimal_sep='.'):
    str_series = series.astype('str')
    clean_series = str_series.str.extract(r'([0-9'+decimal_sep+']+)')
    clean_series = clean_series.astype('float')
    return clean_series

def colType2Int(series, decimal_sep='.'):
    str_series = series.astype('str')
    clean_series = str_series.str.extract(r'([0-9'+decimal_sep+']+)')
    clean_series = clean_series.astype('float')
    clean_series = clean_series.astype('int')
    return clean_series

# Cast certain columns to float after cleaning
float_cols = ['baseLastPrice','strikePrice','bidPrice','midpoint','askPrice','lastPrice','volumeOpenInterestRatio','volatility']
for col in float_cols:
    df[col] = colType2Float(df[col])

# Changing date columns to datetime
date_cols = ['expirationDate','tradeTime','exportedAt']
for col in date_cols:
    df[col] = pd.to_datetime(df[col])

# Cast columns to int
int_cols = ['volume','openInterest']
for col in int_cols:
    df[col] = colType2Int(df[col])

# add target label
# stock price reached strike price at expiration date

# Using above functions
contracts_prices = getContractPrices(df)

# Filter df on only short time to expiration
df = limitDaysToExpiration(df)
df_enr = df.merge(contracts_prices, on=['baseSymbol','expirationDate','exportedAt'])

# Save enriched df as csv
df_enr.to_csv('data/barchart_yf_enr_1.csv')

